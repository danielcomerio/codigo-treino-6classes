import pandas as pd
import numpy as np
import torch
import os

import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
import datetime
import time

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from settings import DATASET_PNG_PATH, DATASET_TEXT_PATH, TRAINED_MODELS_PATH, MODEL_INFO_PATH, EPOCHS, LEARNING_RATE, BATCH_SIZE
from rsnaDatasetImage import generate_transforms, generate_dataset, generate_loader
from work_with_text_file import get_files_name
from viewer import save_graph_train_val
from models import initialize_pretrained_model


def main(model_name):
    files_list = ["train.csv", "val.csv"]
    train_fileNames, val_fileNames = get_files_name(
        DATASET_TEXT_PATH, files_list)
    
    
    dataset_lines = pd.read_csv(os.path.join(
        DATASET_TEXT_PATH, "all_data.csv"), delimiter=';')


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)


    folders_to_create = ["models_graph", "epoch_metrics"]
    for fold in folders_to_create:
        path = os.path.join(MODEL_INFO_PATH, fold)
        if not os.path.isdir(path):
            os.makedirs(path)

    if not os.path.isdir(TRAINED_MODELS_PATH):
        os.makedirs(TRAINED_MODELS_PATH)


    #model_name_list = ["squeezenet", "densenet121", "densenet169", "resnext50", "vgg16"]
    trains_info_txt = open(os.path.join(
        MODEL_INFO_PATH, "trainings_duration.txt"), 'w')
    #for model_name in model_name_list:
    epoch_train_metrics_txt = open(os.path.join(
        MODEL_INFO_PATH, "epoch_metrics", f"{model_name}.txt"), 'w')

    # Initialize the model for this run
    net, input_size = initialize_pretrained_model(model_name, 6, use_pretrained=True)
    net = net.to(device)


    train_transform, val_transform = generate_transforms(input_size)

    train_dataset, val_dataset = generate_dataset(DATASET_PNG_PATH,
                                                    dataset_lines,
                                                    train_fileNames,
                                                    train_transform,
                                                    val_fileNames,
                                                    val_transform,
                                                    True) # return_all_labels
    
    train_data, val_data = generate_loader(train_dataset,
                                            BATCH_SIZE,
                                            val_dataset,
                                            BATCH_SIZE,
                                            2)


    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    num_epochs = EPOCHS


    train_losses, train_hemo_accuracies = [], []
    val_losses, val_hemo_accuracies = [], []

    best_val_f1 = -1
    best_epoch = -1
    
    start_time = time.time()
    # train loop
    for epoch in range(num_epochs):
        epoch_time = time.time()
        train_loss = 0
        train_hemo_hits = 0

        preds = []
        trues = []
        
        dict_hemo = {"epidural":         {"predicted": [], "labels": []},
                        "intraparenchymal": {"predicted": [], "labels": []},
                        "intraventricular": {"predicted": [], "labels": []},
                        "subarachnoid":     {"predicted": [], "labels": []},
                        "subdural":         {"predicted": [], "labels": []},
                        "any":              {"predicted": [], "labels": []}
                        }

        net.train()
        for batch_idx, (name, img, label) in tqdm(enumerate(train_data)):
            img = img.to(device)
            #label = label.to(device)
            optimizer.zero_grad()  # reinitialize gradients
            logits = net(img)

            if model_name == "inception":
                logits = logits[0]

            loss = 0
            for i in range(len(logits)):
                currently_label = [item[i] for item in label]
                loss += loss_fn(logits[i], torch.Tensor(currently_label).to(device))

                pred = [(1 if elem >= 0.5 else 0) for elem in logits[i]]

                # for pos, hemo in enumerate(dict_hemo):
                #     dict_hemo[hemo]["predicted"].append(pred[pos])
                #     dict_hemo[hemo]["labels"].append(currently_label[pos])
                
                train_hemo_hits += 1 if pred[5] == currently_label[5] else 0 # any
                
            loss.backward()  # compute gradients in a backward pass
            optimizer.step()  # update weights
            train_loss += loss
            
        train_loss /= len(train_dataset)
        train_losses.append(train_loss)
        train_hemo_acc = train_hemo_hits / len(train_dataset)
        train_hemo_accuracies.append(train_hemo_acc)

        val_loss = 0
        val_hemo_hits = 0
        val_frac_hits = 0
        val_preds = []
        val_trues = []

        net.eval()
        with torch.no_grad():
            for batch_idx, (name, img, label) in tqdm(enumerate(val_data)):
                img = img.to(device)
                #label = label.to(device)
                logits = net(img)
                
                loss = 0
                for i in range(len(logits)):
                    currently_label = [item[i] for item in label]
                    currently_label_int = [int(elem) for elem in currently_label]
                    loss += loss_fn(logits[i], torch.Tensor(currently_label).to(device))

                    pred = [(1 if elem >= 0.5 else 0) for elem in logits[i]]

                    for pos, hemo in enumerate(dict_hemo):
                        dict_hemo[hemo]["predicted"].append(pred[pos])
                        dict_hemo[hemo]["labels"].append(currently_label_int[pos])

                    val_hemo_hits += 1 if int(pred[5]) == int(currently_label_int[5]) else 0 # any
                
                val_loss += loss

        total_epoch_time = time.time() - epoch_time
        print(f"\nDuração da época {epoch}: {total_epoch_time}\n\n")

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        val_hemo_acc = val_hemo_hits / len(val_dataset)
        val_hemo_accuracies.append(val_hemo_acc)

        val_preds_any = dict_hemo["any"]["predicted"]
        val_trues_any = dict_hemo["any"]["labels"]
        val_any_f1 = f1_score(val_trues_any, val_preds_any)

        if val_any_f1 > best_val_f1:
            torch.save(net, os.path.join(
                TRAINED_MODELS_PATH, f"{model_name}.pt"))
            print("Saved best f1 model (any class).")
            best_val_f1 = val_any_f1
            best_epoch = epoch

        val_hemo_precision = precision_score(val_trues_any, val_preds_any)
        val_hemo_recall = recall_score(val_trues_any, val_preds_any)

        print(f"epoch: {epoch} | loss: {train_loss:.3f}, hemo_acc: {train_hemo_acc:.3f} || val_loss: {val_loss:.3f}, val_hemo_acc: {val_hemo_acc:.3f}, val_hemo_precision: {val_hemo_precision:.3f}, val_hemo_recall: {val_hemo_recall:.3f}, val_any_f1: {val_any_f1:.3f}")

        # print("train confusion matrix:")
        # print(confusion_matrix(trues, preds))

        epoch_train_metrics_txt.write(
            f"epoch: {epoch} | loss: {train_loss:.3f}, hemo_acc: {train_hemo_acc:.3f} || val_loss: {val_loss:.3f}, val_hemo_acc: {val_hemo_acc:.3f}, val_hemo_precision: {val_hemo_precision:.3f}, val_hemo_recall: {val_hemo_recall:.3f}, val_any_f1: {val_any_f1:.3f}")

        # epoch_train_metrics_txt.write("train confusion matrix:\n")
        # [epoch_train_metrics_txt.write(str(row) + '\n')
        #  for row in confusion_matrix(trues, preds)]

        print()
        print("Validation confusion matrix")
        print()
        epoch_train_metrics_txt.write("Validation confusion matrix\n\n")
        for pos, hemo in enumerate(dict_hemo):
            val_preds = dict_hemo[hemo]["predicted"]
            val_trues = dict_hemo[hemo]["labels"]
            
            print(f"{hemo} confusion matrix:")
            print(confusion_matrix(val_trues, val_preds))
            print()
            print()

            epoch_train_metrics_txt.write(f"{hemo} confusion matrix:\n")
            [epoch_train_metrics_txt.write(
                str(row) + '\n') for row in confusion_matrix(val_trues, val_preds)]
            epoch_train_metrics_txt.write('\n')
            epoch_train_metrics_txt.write('\n')
        
        epoch_train_metrics_txt.write('\n\n\n\n')

    total_time_train = time.time() - start_time
    trains_info_txt.write(
        f'Tempo total de treino da rede "{model_name}" -> {str(datetime.timedelta(seconds=total_time_train))}\n\n\n')

    print(f'Tempo total de treino da rede "{model_name}" -> {str(datetime.timedelta(seconds=total_time_train))}\n\n')
    print("----------------\n\n\n\n")
    
    epoch_train_metrics_txt.write(f"\n\n\n\nBest Epoch: {best_epoch}\n\n")
    epoch_train_metrics_txt.close()


# 11 Step
    save_graph_train_val(os.path.join(MODEL_INFO_PATH, "models_graph"), model_name,
                            train_losses, val_losses, train_hemo_accuracies, val_hemo_accuracies)

    trains_info_txt.close()

    return 0


if __name__ == "__main__":
    main()

import pandas as pd
import torch
import os

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, classification_report

from settings import DATASET_TEXT_PATH, DATASET_PNG_PATH, TRAINED_MODELS_PATH, MODEL_INFO_PATH, BATCH_SIZE
from rsnaDatasetImage import generate_transforms, generate_dataset, generate_loader
from work_with_text_file import get_files_name
from models import initialize_pretrained_model


def evaluate_image(model,
                   device,
                   test_loader,
                   ):
    dict_hemo = {"epidural":        {"predicted": [], "labels": []},
                "intraparenchymal": {"predicted": [], "labels": []},
                "intraventricular": {"predicted": [], "labels": []},
                "subarachnoid":     {"predicted": [], "labels": []},
                "subdural":         {"predicted": [], "labels": []},
                "any":              {"predicted": [], "labels": []}
                }
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (names, images, labels) in tqdm(enumerate(test_loader)):
            images = images.to(device)
            logits = model(images)
            logits = logits.to("cpu")

            for i in range(len(logits)):
                currently_label = [item[i] for item in labels]

                pred = [(1 if elem >= 0.5 else 0) for elem in logits[i]]

                for pos, hemo in enumerate(dict_hemo):
                    dict_hemo[hemo]["predicted"].append(pred[pos])
                    dict_hemo[hemo]["labels"].append(currently_label[pos])
            
    return dict_hemo


def main(model_name):
    path = os.path.join(MODEL_INFO_PATH, "test_metrics")
    if not os.path.isdir(path):
        os.makedirs(path)
    
    #models_name = ["squeezenet", "resnet", "vgg", "densenet", "inception"] # inception trained in 13 epochs
    #models_name = ["densenet"]
    
    #for model_name in models_name:
    test_metrics_txt = open(os.path.join(
        MODEL_INFO_PATH, "test_metrics", f"{model_name}.txt"), 'w')
    
    _, input_size = initialize_pretrained_model(model_name, 6)

    model_path = os.path.join(TRAINED_MODELS_PATH, f"{model_name}.pt")
    model = torch.load(model_path) # , map_location=torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)

    [test_fileNames] = get_files_name(DATASET_TEXT_PATH, ["test.csv"])

    dataset_lines = pd.read_csv(
        DATASET_TEXT_PATH.joinpath("all_data.csv"),
        delimiter=';')

    (test_transform) = generate_transforms(
        input_size, is_train=False)

    (test_dataset) = generate_dataset(
        DATASET_PNG_PATH,
        dataset_lines,
        test_fileNames,
        test_transform,
        return_all_labels=True,
        is_train=False)

    (test_loader) = generate_loader(
        test_dataset,
        BATCH_SIZE,
        workers=2,
        is_train=False)

    dict_hemo = evaluate_image(model, device, test_loader)

    print()
    print("Test confusion matrix")
    print()
    test_metrics_txt.write("Test confusion matrix\n\n")
    for pos, hemo in enumerate(dict_hemo):
        val_preds = dict_hemo[hemo]["predicted"]
        val_trues = dict_hemo[hemo]["labels"]
        
        print(f"{hemo} metrics:")
        print(classification_report(val_trues, val_preds))
        print(f"{hemo} confusion matrix:")
        print(confusion_matrix(val_trues, val_preds))
        print()
        print()

        test_metrics_txt.write(f"{hemo} metrics:\n")
        test_metrics_txt.write(classification_report(val_trues, val_preds))
        test_metrics_txt.write('\n')
        
        test_metrics_txt.write(f"{hemo} confusion matrix:\n")
        [test_metrics_txt.write(
            str(row) + '\n') for row in confusion_matrix(val_trues, val_preds)]
        test_metrics_txt.write('\n\n\n\n\n')
        
    test_metrics_txt.close()

    return 0


if __name__ == "__main__":
    main()

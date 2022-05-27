import matplotlib.pyplot as plt
from pathlib import Path
import os


def save_graph_train_val(path_save: Path,
                         graph_name: str,
                         train_losses: None, # list[list[float]]
                         val_losses: None, # list[list[float]]
                         train_hemo_accuracies: None, # list[list[float]]
                         val_hemo_accuracies: None) -> None: # list[list[float]]
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([number.tolist()
             for number in train_losses], label='train')
    plt.plot([number.tolist()
              for number in val_losses], label='val')

    plt.legend()
    plt.title("loss")

    plt.subplot(1, 2, 2)
    plt.plot([number#.tolist()
             for number in train_hemo_accuracies], label='train')
    plt.plot([number#.tolist()
             for number in val_hemo_accuracies], label='val')

    plt.legend()
    plt.title("hemo accuracy")

    graph_name = f"graph_{graph_name}.png"
    plt.savefig(os.path.join(path_save, graph_name), bbox_inches='tight')
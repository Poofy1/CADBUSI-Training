import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np


    
def plot_loss(train_losses, valid_losses, save_path):
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(valid_losses, label='Validation Loss', color='red')

    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(save_path)


def plot_Confusion(all_targs, all_preds, vocab, save_path):
    # Generate the confusion matrix
    cm = confusion_matrix(all_targs, all_preds)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # Use the passed vocab
    tick_marks = np.arange(len(vocab))

    plt.xticks(tick_marks, vocab, rotation=45)
    plt.yticks(tick_marks, vocab)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(save_path)
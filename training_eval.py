import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

    
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



def plot_Confusion(all_targs, all_preds, vocab, file_path):
    # Compute the confusion matrix
    cm = confusion_matrix(all_targs, all_preds)
    
    # Normalize the confusion matrix (optional)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the confusion matrix
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap='Blues', xticklabels=vocab, yticklabels=vocab, ax=ax)
    
    # Set labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    # Save the figure
    fig.savefig(file_path)
    plt.close(fig)
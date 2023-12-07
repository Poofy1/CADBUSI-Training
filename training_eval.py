import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, roc_curve, auc
import numpy as np
from fastai.vision.all import *
import seaborn as sns
from itertools import cycle

    
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
    plt.close()

def save_accuracy_to_file(epoch, train_acc, val_acc, label_columns, file_path):
    with open(file_path, 'a') as file:
        # Write the epoch number
        file.write(f"Epoch {epoch + 1}\n")

        # Write training accuracies
        file.write("Training Accuracies:\n")
        for idx, acc in enumerate(train_acc):
            file.write(f"  {label_columns[idx]}: {acc:.4f}\n")

        # Write validation accuracies
        file.write("Validation Accuracies:\n")
        for idx, acc in enumerate(val_acc):
            file.write(f"  {label_columns[idx]}: {acc:.4f}\n")

        file.write("\n")  # Add a newline for separation between epochs


def plot_single_roc_curve(all_tpr, all_fpr, save_path):
    plt.figure(figsize=(10, 6))
    
    # Plot ROC curve for each epoch
    for i in range(len(all_tpr)):
        plt.plot(all_fpr[i], all_tpr[i], label=f'ROC curve at Epoch {i+1}')

    # Add a red dotted diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random (AUC = 0.5)')

    plt.title('ROC Curve Over Epochs')
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.legend(loc="lower right")

    plt.savefig(save_path)
    plt.close()


def plot_multi_roc_curve(all_fpr, all_tpr, n_classes, save_path):
    plt.figure(figsize=(10, 6))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'yellow', 'purple', 'pink', 'grey', 'olive', 'cyan'])
    
    # Iterate over classes
    for i, color in zip(range(n_classes), colors):
        # Plot ROC curve for each epoch for this class
        for j in range(len(all_tpr[i])):
            plt.plot(all_fpr[i][j], all_tpr[i][j], color=color, alpha=0.3, label=f'Class {i} Epoch {j+1}' if j == 0 else "")

    # Add a red dotted diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random (AUC = 0.5)')

    plt.title('Multi-Class ROC Curve Over Epochs')
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.legend(loc="lower right")

    plt.savefig(save_path)
    plt.close()
    
    

def save_state(e, label_columns, train_acc, val_loss, val_acc, model_folder, model_name, bagmodel, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs):
    
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"
    
    # Calculate current epoch metrics
    all_preds_np = all_preds.numpy() if isinstance(all_preds, torch.Tensor) else np.array(all_preds)
    all_targs_np = all_targs.numpy() if isinstance(all_targs, torch.Tensor) else np.array(all_targs)
    
    n_classes = all_preds_np.shape[1] if len(all_preds_np.shape) > 1 else 1
    
    # Save the model and optimizer
    torch.save(bagmodel.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)

    # Initialize or load previous metrics
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            all_fpr = stats.get('all_fpr', [])
            all_tpr = stats.get('all_tpr', [])
    else:
        all_fpr = []
        all_tpr = []
    
    
    # Calculate ROC curve differently based on the type of classification
    if n_classes == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(all_targs_np, all_preds_np)
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        plot_single_roc_curve(fpr, tpr, f"{model_folder}/{model_name}_roc.png")

    
    
    # Save updated stats with all_fpr and all_tpr
    with open(stats_path, 'wb') as f:
        pickle.dump({
            'epoch': e + 1,
            'train_losses': train_losses_over_epochs,
            'valid_losses': valid_losses_over_epochs,
            'val_loss': val_loss,
            'all_fpr': all_fpr,
            'all_tpr': all_tpr
        }, f)

    # Save the plots
    plot_loss(train_losses_over_epochs, valid_losses_over_epochs, f"{model_folder}/{model_name}_loss.png")
    save_accuracy_to_file(e, train_acc, val_acc, label_columns, f"{model_folder}/{model_name}_accuracy.txt")

    # Save the confusion matrix
    if n_classes == 1:
        vocab = ['not malignant', 'malignant']
        plot_Confusion(all_targs, all_preds, vocab, f"{model_folder}/{model_name}_confusion.png")
    
    

def plot_Confusion(all_targs, all_preds, vocab, file_path):
    # Compute the confusion matrix
    cm = confusion_matrix(all_targs, all_preds)
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=vocab, yticklabels=vocab, ax=ax)
    
    # Invert the y-axis to make it display correctly
    ax.invert_yaxis()

    # Set labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Validation Confusion Matrix')

    # Save the figure
    fig.savefig(file_path)
    plt.close(fig)

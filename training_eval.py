import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, roc_curve, auc
import numpy as np
from fastai.vision.all import *
import seaborn as sns

    
def plot_loss(train_losses, valid_losses, train_acc, val_acc, save_path):
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(valid_losses, label='Validation Loss', color='red')

    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Position for the text, feel free to adjust these values as needed
    x_position = len(train_losses) * 0.1
    y_position = max(max(train_losses), max(valid_losses))

    # Add text Training and Validation Accuracy
    plt.text(x_position, y_position*0.925, f'Train ACC: {train_acc:.4f}', fontsize=9)
    plt.text(x_position, y_position*0.9, f'Val ACC: {val_acc:.4f}', fontsize=9)

    # Save the plot as a PNG file
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(all_tpr, all_fpr, save_path):
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


def save_state(e, train_acc, val_acc, model_folder, model_name, bagmodel, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs):
    
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"

    # Initialize or load previous metrics
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            all_fpr = stats.get('all_fpr', [])
            all_tpr = stats.get('all_tpr', [])
    else:
        all_fpr = []
        all_tpr = []

    # Calculate current epoch metrics
    all_preds_np = all_preds.numpy() if isinstance(all_preds, torch.Tensor) else np.array(all_preds)
    all_targs_np = all_targs.numpy() if isinstance(all_targs, torch.Tensor) else np.array(all_targs)
    fpr, tpr, _ = roc_curve(all_targs_np, all_preds_np)
    all_fpr.append(fpr)
    all_tpr.append(tpr)

    # Save the model and optimizer
    torch.save(bagmodel.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save updated stats with all_fpr and all_tpr
    with open(stats_path, 'wb') as f:
        pickle.dump({
            'epoch': e + 1,
            'train_losses': train_losses_over_epochs,
            'valid_losses': valid_losses_over_epochs,
            'val_acc': val_acc,
            'all_fpr': all_fpr,
            'all_tpr': all_tpr
        }, f)

    # Save the plots
    plot_loss(train_losses_over_epochs, valid_losses_over_epochs, train_acc, val_acc, f"{model_folder}/{model_name}_loss.png")
    plot_roc_curve(all_tpr, all_fpr, f"{model_folder}/{model_name}_roc.png")

    
    # Save the confusion matrix
    vocab = ['not malignant', 'malignant']  # Replace with your actual vocab
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

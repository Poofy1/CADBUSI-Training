import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
import numpy as np
from fastai.vision.all import *
import seaborn as sns

    
def plot_loss(train_losses, valid_losses, auc, sensitivity, specificity, train_acc, val_acc, save_path):
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

    # Add text annotations for AUC, Sensitivity, Specificity, Training and Validation Accuracy
    plt.text(x_position, y_position, f'AUC: {auc:.4f}', fontsize=9)
    plt.text(x_position, y_position*0.975, f'Sensitivity: {sensitivity:.4f}', fontsize=9)
    plt.text(x_position, y_position*0.95, f'Specificity: {specificity:.4f}', fontsize=9)
    plt.text(x_position, y_position*0.925, f'Train ACC: {train_acc:.4f}', fontsize=9)
    plt.text(x_position, y_position*0.9, f'Val ACC: {val_acc:.4f}', fontsize=9)

    # Save the plot as a PNG file
    plt.savefig(save_path)
    plt.close()


def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    return specificity

def save_state(e, train_acc, val_acc, model_folder, model_name, bagmodel, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs):
    
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"
    
    # Assuming all_preds and all_targs are numpy arrays or can be converted to them
    all_preds_np = all_preds.numpy() if isinstance(all_preds, torch.Tensor) else np.array(all_preds)
    all_targs_np = all_targs.numpy() if isinstance(all_targs, torch.Tensor) else np.array(all_targs)

    # Calculate additional metrics
    auc = roc_auc_score(all_targs_np, all_preds_np)
    sensitivity = recall_score(all_targs_np, all_preds_np)
    specificity = calculate_specificity(all_targs_np, all_preds_np)
    
    # Save the model
    torch.save(bagmodel.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # Save stats
    with open(stats_path, 'wb') as f:
        pickle.dump({
            'epoch': e + 1,  # Save the next epoch to start
            'train_losses': train_losses_over_epochs,
            'valid_losses': valid_losses_over_epochs,
            'val_acc': val_acc,  # Save the validation accuracy
            'auc': auc,  # Save the AUC
            'sensitivity': sensitivity,  # Save the Sensitivity (Recall)
            'specificity': specificity,  # Save the Specificity
        }, f)


    # Save the loss graph
    plot_loss(train_losses_over_epochs, valid_losses_over_epochs, auc, sensitivity, specificity, train_acc, val_acc, f"{model_folder}/{model_name}_loss.png")
    
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

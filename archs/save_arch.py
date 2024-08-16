import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
from fastai.vision.all import *
import seaborn as sns
from itertools import cycle
import shutil
    
def plot_loss(train_losses, valid_losses, save_path):
    
    if not train_losses or not valid_losses:
        print("No loss data available to plot.")
        return
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(valid_losses, label='Validation Loss', color='red')

    plt.title('Bag Loss')
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

        # Check if train_acc is a single float or an iterable of floats
        if isinstance(train_acc, float):
            # If train_acc is a single float, write it directly
            file.write("Training Accuracy:\n")
            file.write(f"  Overall: {train_acc:.4f}\n")
        else:
            # If train_acc is an iterable, write each accuracy
            file.write("Training Accuracies:\n")
            for idx, acc in enumerate(train_acc):
                file.write(f"  {label_columns[idx]}: {acc:.4f}\n")

        # Do the same check for val_acc
        if isinstance(val_acc, float):
            file.write("Validation Accuracy:\n")
            file.write(f"  Overall: {val_acc:.4f}\n")
        else:
            file.write("Validation Accuracies:\n")
            for idx, acc in enumerate(val_acc):
                file.write(f"  {label_columns[idx]}: {acc:.4f}\n")

        file.write("\n")  # Add a newline for separation between epochs


def plot_single_roc_curve(all_fpr, all_tpr, save_path):
    plt.figure(figsize=(10, 6))
    
    # Check if all_fpr and all_tpr are lists of lists for multiple epochs or single lists for one epoch
    if isinstance(all_fpr[0], list):
        # Multiple epochs: Iterate over each epoch
        for i in range(len(all_tpr)):
            plt.plot(all_fpr[i], all_tpr[i], label=f'ROC curve at Epoch {i+1}')
    else:
        # Single epoch: Directly plot
        plt.plot(all_fpr, all_tpr, label='ROC curve')

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
    
    
def plot_Confusion(all_targs, all_preds, vocab, file_path):
    # Compute the confusion matrix
    cm = confusion_matrix(all_targs, all_preds)
    
    # Normalize the confusion matrix by the total number of predictions
    cm_normalized = cm.astype('float') / cm.sum()
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt=".3f", cmap='Blues', xticklabels=vocab, yticklabels=vocab, ax=ax)
    
    # Invert the y-axis to make it display correctly
    ax.invert_yaxis()

    # Set labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Validation Confusion Matrix')

    # Save the figure
    fig.savefig(file_path)
    plt.close(fig)

def save_state(e, label_columns, train_acc, val_loss, val_acc, model_folder, model_name, bagmodel, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs, classifier=None, palm = None):
    model_path = f"{model_folder}/model.pth"
    classifier_path = f"{model_folder}/classifier.pth"
    optimizer_path = f"{model_folder}/optimizer.pth"
    stats_path = f"{model_folder}/stats.pkl"

    # Calculate current epoch metrics
    all_preds_np = all_preds.numpy() if isinstance(all_preds, torch.Tensor) else np.array(all_preds)
    all_targs_np = all_targs.numpy() if isinstance(all_targs, torch.Tensor) else np.array(all_targs)

    n_classes = all_preds_np.shape[1] if len(all_preds_np.shape) > 1 else 1

    # Save the model and optimizer
    torch.save(bagmodel.state_dict(), model_path)
    if classifier:
        torch.save(classifier.state_dict(), classifier_path)
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

    # Check if all_targs and all_preds are not empty before calculating ROC curve
    if len(all_targs) > 0 and len(all_preds) > 0:
        # Calculate ROC curve differently based on the type of classification
        if n_classes == 1:
            # Binary classification
            fpr, tpr, _ = roc_curve(all_targs_np, all_preds_np)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            plot_single_roc_curve(fpr, tpr, f"{model_folder}/roc.png")

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

    if palm is not None:
        # Duplicate head palm state
        palm_state_source = os.path.join(os.path.dirname(model_folder), 'palm_state.pkl')
        palm_state_destination = os.path.join(model_folder, 'palm_state.pkl')
        
        if os.path.exists(palm_state_source) and not os.path.exists(palm_state_destination):
            shutil.copy2(palm_state_source, palm_state_destination)
        elif not os.path.exists(palm_state_source):
            print(f"Warning: palm_state.pkl not found in the parent directory of {model_folder}")
        else:
            print(f"palm_state.pkl already exists in {model_folder}")
            
    # Save the plots
    plot_loss(train_losses_over_epochs, valid_losses_over_epochs, f"{model_folder}/loss.png")
    save_accuracy_to_file(e, train_acc, val_acc, label_columns, f"{model_folder}/accuracy.txt")

    # Save the confusion matrix if all_targs and all_preds are not empty
    if len(all_targs) > 0 and len(all_preds) > 0 and n_classes == 1:
        vocab = ['not malignant', 'malignant']
        plot_Confusion(all_targs, all_preds, vocab, f"{model_folder}/confusion.png")
        
        
        

def setup_model(model, optimizer, config):
    """
    Set up the model, handle loading/saving, and manage configurations.
    
    :param model: The model to set up
    :param optimizer: The optimizer for the model
    :param config: A dictionary containing all necessary configuration parameters
    :return: A dictionary containing the state of the model setup
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)
    
    dataset_name = config['dataset_name']
    arch = config['arch']
    model_name = config['model_version']
    pretrained_name = f"Head_{config['head_name']}_{arch}"
    
    head_folder = os.path.join(parent_dir, "models", pretrained_name)
    head_path = os.path.join(head_folder, f"{pretrained_name}.pth")

    model_folder = os.path.join(parent_dir, "models", pretrained_name, model_name)
    model_path = os.path.join(model_folder, f"model.pth")
    stats_path = os.path.join(model_folder, f"stats.pkl")

    state = {
        'optimizer': optimizer,
        'head_folder': head_folder,
        'pretrained_name': pretrained_name,
        'model_folder': model_folder,
        'model_name': model_name,
        'palm_path': None,
        'train_losses': [],
        'valid_losses': [],
        'epoch': 0,
        'val_acc_best': 0,
        'val_loss_best': 99999,
        'selection_mask': [],
        'warmup': False,
        'pickup_warmup': False
    }

    if os.path.exists(model_path):
        print(f"Loaded pre-existing model from {model_name}")
        encoder_state_dict = torch.load(model_path)
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_state_dict)
        state['train_losses'], state['valid_losses'], state['epoch'], state['val_acc_best'], state['selection_mask'] = load_state(stats_path, model_folder)
        state['palm_path'] = os.path.join(model_folder, "palm_state.pkl")
    else:
        print(f"{model_name} does not exist, creating new instance")
        os.makedirs(model_folder, exist_ok=True)
        state['palm_path'] = os.path.join(head_folder, "palm_state.pkl")
        
        if os.path.exists(head_path):
            state['pickup_warmup'] = True
            encoder_state_dict = torch.load(head_path)
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items() if k.startswith('encoder.')}
            model.encoder.load_state_dict(encoder_state_dict)
            print(f"Loaded pre-trained model from {pretrained_name}")
        else:
            state['warmup'] = True
            os.makedirs(head_folder, exist_ok=True)

        save_config(config, model_folder)
        save_model_architecture(model, model_folder)
        
    
    
    return model, optimizer, state


def load_model_config(folder_path):
    """
    Load the configuration for a specific model.
    """

    # Construct the path to the config file
    config_path = os.path.join(folder_path, "config.json")
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No configuration file found for model {folder_path}")
    
    # Load and return the configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def save_config(config, folder):
    config_path = os.path.join(folder, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def save_model_architecture(model, folder):
    with open(os.path.join(folder, 'model_architecture.txt'), 'w') as f:
        print(model, file=f)

def load_state(stats_path, target_folder):
    selection_mask = []
    
    with open(stats_path, 'rb') as f:
        saved_stats = pickle.load(f)
        train_losses = saved_stats['train_losses']
        valid_losses = saved_stats['valid_losses']
        epoch = saved_stats['epoch']
        val_loss_best = saved_stats['val_loss']
    
    # Load the selection_mask dictionary from the file
    if os.path.exists(f'{target_folder}/selection_mask.pkl'):
        with open(f'{target_folder}/selection_mask.pkl', 'rb') as file:
            selection_mask = pickle.load(file)
            
    return train_losses, valid_losses, epoch, val_loss_best, selection_mask
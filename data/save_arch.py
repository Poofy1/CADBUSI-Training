import matplotlib.pyplot as plt
from fastai.vision.all import *
import shutil
from storage_adapter import * 

    
def plot_loss(train_losses, valid_losses, save_path):
    
    if not train_losses or not valid_losses:
        print("No loss data available to plot.")
        return
    
    train_losses_np = [loss.detach().numpy() if torch.is_tensor(loss) else loss for loss in train_losses]
    valid_losses_np = [loss.detach().numpy() if torch.is_tensor(loss) else loss for loss in valid_losses]

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_np, label='Training Loss', color='blue')
    plt.plot(valid_losses_np, label='Validation Loss', color='red')

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
        if isinstance(train_acc, (float, int)):
            # If train_acc is a single float, write it directly
            file.write("Training Accuracy:\n")
            file.write(f"  Overall: {train_acc:.4f}\n")
        else:
            # If train_acc is an iterable, write each accuracy
            file.write("Training Accuracies:\n")
            for idx, acc in enumerate(train_acc):
                file.write(f"  {label_columns[idx]}: {acc:.4f}\n")

        # Do the same check for val_acc
        if isinstance(val_acc, (float, int)):
            file.write("Validation Accuracy:\n")
            file.write(f"  Overall: {val_acc:.4f}\n")
        else:
            file.write("Validation Accuracies:\n")
            for idx, acc in enumerate(val_acc):
                file.write(f"  {label_columns[idx]}: {acc:.4f}\n")

        file.write("\n")  # Add a newline for separation between epochs

    


def save_state(state, config, train_acc, val_loss, val_acc, bagmodel, optimizers=None, classifier=None, palm=None):
    if state['warmup']:
        model_folder = state['head_folder']
    else:
        model_folder = state['model_folder']
        
    model_path = f"{model_folder}/model.pth"
    classifier_path = f"{model_folder}/classifier.pth"
    stats_path = f"{model_folder}/stats.pkl"
    
    label_columns = config['label_columns']
    train_losses_over_epochs = state['train_losses']
    valid_losses_over_epochs = state['valid_losses']

    # Save the model
    torch.save(bagmodel.state_dict(), model_path)
    
    # Save classifier if provided
    if classifier:
        torch.save(classifier.state_dict(), classifier_path)
    
    # Save optimizers if provided
    if optimizers:
        if isinstance(optimizers, dict):
            # Handle dictionary of optimizers
            for opt_name, optimizer in optimizers.items():
                torch.save(optimizer.state_dict(), f"{model_folder}/{opt_name}.pth")
        else:
            # Handle single optimizer for backward compatibility
            torch.save(optimizers.state_dict(), f"{model_folder}/optimizer.pth")

    # Initialize or load previous metrics
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
            all_fpr = stats.get('all_fpr', [])
            all_tpr = stats.get('all_tpr', [])
    else:
        all_fpr = []
        all_tpr = []

    # Save updated stats with all_fpr and all_tpr
    with open(stats_path, 'wb') as f:
        val_loss_key = 'val_loss_instance' if state['mode'] == 'instance' else 'val_loss_bag'
        pickle.dump({
            'epoch': state['epoch'] + 1,
            'train_losses': train_losses_over_epochs,
            'valid_losses': valid_losses_over_epochs,
            val_loss_key: val_loss,
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
    save_accuracy_to_file(state['epoch'], train_acc, val_acc, label_columns, f"{model_folder}/{state['mode']}_accuracy.txt")
        
        

def setup_model(model, config, optimizers=None):
    """
    Set up the model, handle loading/saving, and manage configurations.
    
    :param model: The model to set up
    :param optimizers: Dictionary of optimizers for the model (e.g. {'instance_optimizer': opt1, 'bag_optimizer': opt2})
    :param config: A dictionary containing all necessary configuration parameters
    :return: model, optimizers dictionary, and state dictionary
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)
    
    model_name = config['model_version']
    pretrained_name = f"{config['head_name']}"
    
    encoder_model_name = config['encoder_model_version']
    encoder_pretrained_name = f"{config['encoder_head_name']}"
    
    head_folder = os.path.join(parent_dir, "models", pretrained_name)
    head_path = os.path.join(head_folder, f"model.pth")

    model_folder = os.path.join(parent_dir, "models", pretrained_name, model_name)
    model_path = os.path.join(model_folder, f"model.pth")
    stats_path = os.path.join(model_folder, f"stats.pkl")

    state = {
        'optimizers': optimizers,
        'head_folder': head_folder,
        'pretrained_name': pretrained_name,
        'model_folder': model_folder,
        'model_name': model_name,
        'mode': 'instance',
        'palm_path': None,
        'train_losses': [],
        'valid_losses': [],
        'epoch': 0,
        'val_loss_instance': 99999,
        'val_loss_bag': 99999,
        'selection_mask': [],
        'warmup': False,
        'pickup_warmup': False
    }

    def load_model(model, model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        return model
    
    # Load optimizer state if available
    def load_optimizer_states(optimizers, folder):
        for name, optimizer in optimizers.items():
            optimizer_path = os.path.join(folder, f"{name}.pth")
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path))
                print(f"Loaded optimizer state for {name}")
        return optimizers
    
    def load_encoder_weights(model, encoder_model_path):
        """Load only encoder weights from a pretrained model, ignoring aggregator and projector"""
        if os.path.exists(encoder_model_path):
            print(f"Loading encoder weights from {encoder_model_path}")
            pretrained_state_dict = torch.load(encoder_model_path)
            
            # Get current model's state dict to compare parameter shapes
            current_state_dict = model.state_dict()
            
            print(f"Total parameters in pretrained model: {len(pretrained_state_dict)}")
            print(f"Total parameters in current model: {len(current_state_dict)}")
            
            
            # Filter to only encoder-related parameters
            encoder_weights = {}
            skipped_params = []
            loaded_params = []
            
            for key, value in pretrained_state_dict.items():
                # Only load parameters that start with 'encoder.' 
                if key.startswith('encoder.'):
                    
                    # Check if this parameter exists in current model and has matching shape
                    if key in current_state_dict:
                        if current_state_dict[key].shape == value.shape:
                            encoder_weights[key] = value
                            loaded_params.append(key)
                        else:
                            skipped_params.append(f"{key} (shape mismatch: {current_state_dict[key].shape} vs {value.shape})")
                    else:
                        skipped_params.append(f"{key} (not found in current model)")
            
            # Load only the compatible encoder weights
            if encoder_weights:
                model.load_state_dict(encoder_weights, strict=False)
                
                # Calculate total parameters (not just parameter tensors)
                total_params = sum(p.numel() for p in encoder_weights.values())
                print(f"\nSuccessfully loaded {len(encoder_weights)} parameter tensors")
                print(f"Total encoder parameters loaded: {total_params:,}")
                

                if skipped_params:
                    print(f"\nSkipped {len(skipped_params)} incompatible parameters:")
                    for param in skipped_params[:5]:
                        print(f"  ✗ {param}")
                    if len(skipped_params) > 5:
                        print(f"  ... and {len(skipped_params) - 5} more")
            else:
                print("Warning: No compatible encoder parameters found to load")
                
        else:
            print(f"Warning: Encoder model path {encoder_model_path} does not exist")
        
    if os.path.exists(model_path):
        print(f"Loaded pre-existing model from {model_name}")
        model = load_model(model, model_path)
        state['train_losses'], state['valid_losses'], state['epoch'], state['val_loss_bag'], state['val_loss_instance'], state['selection_mask'] = load_state(stats_path, model_folder)
        state['palm_path'] = os.path.join(model_folder, "palm_state.pkl")
        
        # Load optimizer states if they exist
        if optimizers:
            optimizers = load_optimizer_states(optimizers, model_folder)
    else:
        print(f"{model_name} does not exist, creating new instance")
        os.makedirs(model_folder, exist_ok=True)
        state['palm_path'] = os.path.join(head_folder, "palm_state.pkl")
        
        if os.path.exists(head_path):
            state['pickup_warmup'] = True
            state['warmup'] = False
            model = load_model(model, head_path)
            print(f"Loaded pre-trained model from {pretrained_name}")
            
            # Try to load optimizer states from head folder
            if optimizers:
                optimizers = load_optimizer_states(optimizers, head_folder)
        else:
            state['warmup'] = True
            os.makedirs(head_folder, exist_ok=True)

        save_config(config, model_folder)
        save_config(config, head_folder)
        save_model_architecture(model, head_folder)
        
    # Load encoder weights from different pretrained model if specified
    if encoder_pretrained_name and encoder_pretrained_name.strip():
        encoder_folder = os.path.join(parent_dir, "models", encoder_pretrained_name, encoder_model_name)
        encoder_model_path = os.path.join(encoder_folder, "model.pth")
        load_encoder_weights(model, encoder_model_path)
        
    return model, optimizers, state


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
    
    # Determine storage client   
    StorageClient.get_instance(None, config['bucket'])
    
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
        val_loss_bag = saved_stats.get('val_loss_bag', 99999)
        val_loss_instance = saved_stats.get('val_loss_instance', 99999)
    
    # Load the selection_mask dictionary from the file
    if os.path.exists(f'{target_folder}/selection_mask.pkl'):
        with open(f'{target_folder}/selection_mask.pkl', 'rb') as file:
            selection_mask = pickle.load(file)
            
    return train_losses, valid_losses, epoch, val_loss_bag, val_loss_instance, selection_mask
import os, pickle
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.pseudo_labels import *
from data.instance_loader import *
from loss.palm import PALM
from util.eval_util import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def process_instance_data(instance_labels, instance_pred, features):
    """
    Flattens and filters instance-level data to remove unknown labels.
    
    Args:
        instance_labels (list): Nested list of instance labels
        instance_pred (list): Nested list of instance predictions
        features (torch.Tensor): Feature tensor
        
    Returns:
        tuple: (known_features, known_instance_pred, known_instance_labels)
    """
    # Flatten labels
    flattened_labels = torch.cat([
        torch.cat([
            label.cuda() if isinstance(label, torch.Tensor) 
            else torch.tensor([label]).cuda() 
            for label in bag_labels
        ])
        for bag_labels in instance_labels
    ])
    
    # Flatten predictions
    flattened_pred = torch.cat([
        torch.cat([
            pred.cuda().view(1) if isinstance(pred, torch.Tensor) 
            else torch.tensor([pred], device='cuda')
            for pred in bag_preds
        ])
        for bag_preds in instance_pred
    ])
    
    # Create and apply mask for known labels
    known_mask = flattened_labels != -1
    known_features = features[known_mask]
    known_instance_pred = flattened_pred[known_mask]
    known_instance_labels = flattened_labels[known_mask]
    
    return known_features, known_instance_pred, known_instance_labels
                
                
                
                
def train_bags(model, optimizer, bag_dataloader_train, BCE_loss, device):
    """Train the model on bags"""
    model.train()
    train_bag_logits = {}

    bags_total = 0
    instance_total = 0
    bag_total_correct = 0
    palm_total_correct = 0
    instance_total_correct = 0
    total_epoch_loss = 0
    train_pred = PredictionTracker()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for (images, yb, instance_labels, unique_id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
        optimizer.zero_grad()

        # Handle both padded tensor and list of tensors cases
        if not isinstance(images, list):
            images, yb = images.cuda(), yb.cuda()
        else:
            images = [img.cuda() for img in images]
            yb = yb.cuda()

        # Forward pass
        bag_pred, _, instance_pred, features = model(images, projector = True, pred_on=True)
        bag_pred = bag_pred.cuda()



        # BAG TRAINING
        # Calculate loss and update
        bag_loss = BCE_loss(bag_pred, yb)
        
        
        
        # INSTANCE TRAINING
        # Filter out unknown labels
        known_features, known_instance_pred, known_instance_labels = process_instance_data(
            instance_labels, 
            instance_pred, 
            features)


        if known_features.size(0) > 0:  # Only if we have known labels
            # Calculate losses on known instances
            palm_loss, _ = palm(known_features, known_instance_labels.long())
            instance_loss = BCE_loss(known_instance_pred, known_instance_labels.float())
            
            # Calculate accuracies
            with torch.no_grad():
                palm_predicted_classes, _ = palm.predict(known_features)
                instance_predicted_classes = torch.sigmoid(known_instance_pred) > 0.5
                
                palm_correct = (palm_predicted_classes == known_instance_labels).sum().item()
                instance_correct = (instance_predicted_classes == known_instance_labels).sum().item()
                
                palm_total_correct += palm_correct
                instance_total_correct += instance_correct
                instance_total += known_instance_labels.size(0)
        else:
            palm_loss = torch.tensor(0.0).cuda()
            instance_loss = torch.tensor(0.0).cuda()
        
        
        # Backward pass
        total_loss = bag_loss + palm_loss + instance_loss
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        total_epoch_loss += total_loss
        predicted = (torch.sigmoid(bag_pred) > 0.5).float()
        bags_total += yb.size(0)
        bag_total_correct += (predicted == yb).sum().item()
        train_pred.update(torch.sigmoid(bag_pred), yb, unique_id)

        torch.cuda.empty_cache()

    
    train_loss = total_epoch_loss / bags_total
    train_acc = bag_total_correct / bags_total
    
    palm_train_acc = palm_total_correct / instance_total
    instance_train_acc = instance_total_correct / instance_total
    
    return train_loss, train_acc, palm_train_acc, instance_train_acc, train_pred

def validate_bags(model, bag_dataloader_val, BCE_loss, device):
    """Validate the model on bags"""
    model.eval()
    total_epoch_loss = 0.0
    bags_total = 0
    instance_total = 0
    bag_total_correct = 0
    palm_total_correct = 0
    instance_total_correct = 0
    val_pred = PredictionTracker()

    with torch.no_grad():
        for (images, yb, instance_labels, unique_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)):
            # Handle both padded tensor and list of tensors cases
            if not isinstance(images, list):
                images, yb = images.cuda(), yb.cuda()
            else:
                images = [img.cuda() for img in images]
                yb = yb.cuda()

            # Forward pass
            bag_pred, _, instance_pred, features = model(images, projector = True, pred_on=True)
            bag_pred = bag_pred.cuda()

            # BAGS
            # Calculate loss
            bag_loss = BCE_loss(bag_pred, yb)
            
            # INSTANCES
            # Filter out unknown labels
            known_features, known_instance_pred, known_instance_labels = process_instance_data(
                instance_labels, 
                instance_pred, 
                features)

            if known_features.size(0) > 0:  # Only if we have known labels
                # Calculate losses on known instances
                palm_loss, _ = palm(known_features, known_instance_labels.long())
                instance_loss = BCE_loss(known_instance_pred, known_instance_labels.float())
                
                # Calculate accuracies
                with torch.no_grad():
                    palm_predicted_classes, _ = palm.predict(known_features)
                    instance_predicted_classes = torch.sigmoid(known_instance_pred) > 0.5
                    
                    palm_correct = (palm_predicted_classes == known_instance_labels).sum().item()
                    instance_correct = (instance_predicted_classes == known_instance_labels).sum().item()
                    
                    palm_total_correct += palm_correct
                    instance_total_correct += instance_correct
                    instance_total += known_instance_labels.size(0)
            else:
                palm_loss = torch.tensor(0.0).cuda()
                instance_loss = torch.tensor(0.0).cuda()
                
                
            total_loss = bag_loss + palm_loss + instance_loss
            total_epoch_loss += total_loss

            # Track metrics
            predicted = (torch.sigmoid(bag_pred) > 0.5).float()
            bags_total += yb.size(0)
            bag_total_correct += (predicted == yb).sum().item()
            val_pred.update(torch.sigmoid(bag_pred), yb, unique_id)

            torch.cuda.empty_cache()

    val_loss = total_epoch_loss / bags_total
    val_acc = bag_total_correct / bags_total
    
    palm_val_acc = palm_total_correct / instance_total
    instance_val_acc = instance_total_correct / instance_total
    
    return val_loss, val_acc, palm_val_acc, instance_val_acc, val_pred



if __name__ == '__main__':
    # Config
    model_version = '1'
    head_name = "TEST182"
    data_config = DogDataConfig #FishDataConfig or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = build_model(config)    
    
    # LOSS INIT
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 0, lambda_pcon=1).cuda()
    BCE_loss = nn.BCEWithLogitsLoss()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, config, optimizer)
    palm.load_state(state['palm_path'])

    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        

        iteration = 0
        while True:
            # Train
            train_loss, train_acc, palm_train_acc, instance_train_acc, train_pred = train_bags(model, optimizer, bag_dataloader_train, BCE_loss, device)
            
            # Validate
            val_loss, val_acc, palm_val_acc, instance_val_acc, val_pred = validate_bags(model, bag_dataloader_val, BCE_loss, device)

            state['train_losses'].append(train_loss.detach().cpu().numpy())
            state['valid_losses'].append(val_loss.detach().cpu().numpy())    
            
            iteration+=1
            print(f"[{iteration}] | Bag Acc | Palm Acc | Inst Acc | Loss")
            print(f'train  {train_acc:.5f} | {palm_train_acc:.5f} | {instance_train_acc:.5f} | {train_loss:.4f}')
            print(f'val    {val_acc:.5f} | {palm_val_acc:.5f} | {instance_val_acc:.5f} | {val_loss:.4f}')

            # Save the model if validation loss improves
            if val_loss < state['val_loss_bag']:
                state['val_loss_bag'] = val_loss
                state['mode'] = 'bag'
                target_folder = state['head_folder'] if state['warmup'] else state['model_folder']

                save_state(state, config, train_acc, val_loss, val_acc, model, optimizer)
                save_metrics(config, state, train_pred, val_pred)
                palm.save_state(os.path.join(target_folder, "palm_state.pkl"))
                print("Saved checkpoint due to improved val_loss_bag")

                state['epoch'] += 1
                
                """# Create selection mask
                predictions_ratio = prediction_anchor_scheduler(state['epoch'], config['total_epochs'], 0, config['initial_ratio'], config['final_ratio'])
                state['selection_mask'] = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)"""


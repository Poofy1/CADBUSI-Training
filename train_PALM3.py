import os, pickle
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.sudo_labels import *
from archs.model_MIL import *
from data.instance_loader import *
from loss.palm import PALM
from util.eval_util import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"






def train_instances(model, palm, optimizer, instance_dataloader_train, device, BCE_loss):
    # Training metrics
    losses = AverageMeter()
    palm_total_correct = 0
    instance_total_correct = 0
    total_samples = 0
    train_pred = PredictionTracker()
    
    model.train()
    
    # Training loop
    for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
        images = images.cuda(non_blocking=True)
        instance_labels = instance_labels.cuda(non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        _, _, instance_predictions, features = model(images, projector=True)
        features.to(device)
        
        # Calculate losses
        palm_loss, _ = palm(features, instance_labels.long())
        bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())
        total_loss = palm_loss + bce_loss_value
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses.update(total_loss.item(), images[0].size(0))
        
        # Calculate accuracies
        with torch.no_grad():
            palm_predicted_classes, _ = palm.predict(features)
            instance_predicted_classes = (instance_predictions) > 0.5
            
            palm_correct = (palm_predicted_classes == instance_labels).sum().item()
            instance_correct = (instance_predicted_classes == instance_labels).sum().item()
            
            palm_total_correct += palm_correct
            instance_total_correct += instance_correct
            total_samples += instance_labels.size(0)
            
            train_pred.update(instance_predictions, instance_labels, unique_id)
        
        torch.cuda.empty_cache()

    # Calculate training metrics
    palm_train_acc = palm_total_correct / total_samples
    instance_train_acc = instance_total_correct / total_samples
    
    return palm_train_acc, instance_train_acc, train_pred, losses
    
                
                
def validate_instances(model, palm, instance_dataloader_val, BCE_loss, device):
    """Validate the feature extractor"""
    model.eval()
    palm_total_correct = 0
    instance_total_correct = 0
    total_samples = 0
    val_losses = AverageMeter()
    val_pred = PredictionTracker()
    
    with torch.no_grad():
        for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
            images = images.cuda(non_blocking=True)
            instance_labels = instance_labels.cuda(non_blocking=True)

            # Forward pass
            _, _, instance_predictions, features = model(images, projector=True)
            features.to(device)
            
            # Calculate losses
            palm_loss, _ = palm(features, instance_labels, update_prototypes=False)
            bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())
            total_loss = palm_loss + bce_loss_value
            
            val_losses.update(total_loss.item(), images[0].size(0))

            # Calculate accuracies
            palm_predicted_classes, _ = palm.predict(features)
            instance_predicted_classes = (instance_predictions) > 0.5

            palm_correct = (palm_predicted_classes == instance_labels).sum().item()
            instance_correct = (instance_predicted_classes == instance_labels).sum().item()
            
            palm_total_correct += palm_correct
            instance_total_correct += instance_correct
            total_samples += instance_labels.size(0)
            
            val_pred.update(instance_predictions, instance_labels, unique_id)
            
            torch.cuda.empty_cache()

    # Calculate training metrics
    palm_val_acc = palm_total_correct / total_samples
    instance_val_acc = instance_total_correct / total_samples
    
    return palm_val_acc, instance_val_acc, val_pred, val_losses  
                
def train_bags(model, optimizer, bag_dataloader_train, BCE_loss, device):
    """Train the model on bags"""
    model.train()
    train_bag_logits = {}
    total_loss = 0.0
    total = 0
    correct = 0
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
        bag_pred, _, instance_pred, _ = model(images, pred_on=True)
        bag_pred = bag_pred.cuda()

        # Store instance predictions for pseudo-labeling
        split_sizes = [bag.size(0) for bag in images] if isinstance(images, list) else [images.size(0)]
        y_hat_per_bag = torch.split(instance_pred, split_sizes, dim=0)
        for i, y_h in enumerate(y_hat_per_bag):
            #y_h_sigmoid = torch.sigmoid(y_h)
            train_bag_logits[unique_id[i].item()] = y_h.detach().cpu().numpy()

        # Calculate loss and update
        bag_loss = BCE_loss(bag_pred, yb)
        bag_loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += bag_loss.item() * yb.size(0)
        predicted = (bag_pred > 0.5).float()
        total += yb.size(0)
        correct += (predicted == yb).sum().item()
        train_pred.update(bag_pred, yb, unique_id)

        torch.cuda.empty_cache()

    train_loss = total_loss / total
    train_acc = correct / total
    
    return train_loss, train_acc, train_pred, train_bag_logits

def validate_bags(model, bag_dataloader_val, BCE_loss, device):
    """Validate the model on bags"""
    model.eval()
    total = 0
    correct = 0
    total_val_loss = 0.0
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
            bag_pred, _, _, features = model(images, pred_on=True)
            bag_pred = bag_pred.cuda()

            # Calculate loss
            loss = BCE_loss(bag_pred, yb)
            total_val_loss += loss.item() * yb.size(0)

            # Track metrics
            predicted = (bag_pred > 0.5).float()
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
            val_pred.update(bag_pred, yb, unique_id)

            torch.cuda.empty_cache()

    val_loss = total_val_loss / total
    val_acc = correct / total
    
    return val_loss, val_acc, val_pred



if __name__ == '__main__':
    # Config
    model_version = '1'
    head_name = "TEST181"
    data_config = DogDataConfig #FishDataConfig or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    
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
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
            # Add early stopping parameters
            if state['warmup']:
                target_count = float('inf')  # Will be controlled by early stopping
                patience = 3
                patience_counter = 0
                best_val_loss = float('inf')
            else:
                target_count = config['MIL_train_count']

            print('Training Bags')
            print(f'Warmup Mode: {state["warmup"]}')

            iteration = 0
            while True:
                # Train
                train_loss, train_acc, train_pred, train_bag_logits = train_bags(model, optimizer, bag_dataloader_train, BCE_loss, device)
                
                # Validate
                val_loss, val_acc, val_pred = validate_bags(model, bag_dataloader_val, BCE_loss, device)

                state['train_losses'].append(train_loss)
                state['valid_losses'].append(val_loss)    
                
                print(f"[{iteration+1}/{config['MIL_train_count'] if not state['warmup'] else 'inf'}] | Acc | Loss")
                print(f"Train | {train_acc:.4f} | {train_loss:.4f}")
                print(f"Val | {val_acc:.4f} | {val_loss:.4f}")

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

                # Early stopping logic for warmup mode
                if state['warmup']:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {iteration+1} iterations")
                        break
                
                iteration += 1
                
                # Check if we've reached target count for non-warmup mode
                if not state['warmup'] and iteration >= target_count:
                    break





        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            

        # Used the instance predictions from bag training to update the Instance Dataloader
        instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=True)
        instance_dataset_val = Instance_Dataset(bags_val, [], transform=val_transform, warmup=True)
        train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
        val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], seed=1)
        instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, collate_fn = collate_instance)
        instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
        print('\nTraining Instances')
        for iteration in range(config['feature_extractor_train_count']):
            
            # Train 
            palm_train_acc, instance_train_acc, train_pred, losses = train_instances(model, palm, optimizer, instance_dataloader_train, device, BCE_loss)
            
            # Val               
            palm_val_acc, instance_val_acc, val_pred, val_losses = validate_instances(model, palm, instance_dataloader_val, BCE_loss, device)
            
            print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train Palm Acc: {palm_train_acc:.5f}, Train FC Acc: {instance_train_acc:.5f}')
            print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}, Val Palm Acc: {palm_val_acc:.5f}, Val FC Acc: {instance_val_acc:.5f}')
            
            

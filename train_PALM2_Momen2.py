import os
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.sudo_labels import *
from archs.model_solo_MIL_saliency import *
from data.bag_loader import *
from data.instance_loader import *
from loss.palm import PALM
from util.eval_util import *
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    


if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "TEST143"
    data_config = DogDataConfig  #FishDataConfig or DogDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    
    # LOSS INIT
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 0, lambda_pcon=1).cuda()
    BCE_loss = nn.BCELoss()
    
    # Combine the parameters from both the embedding model and the PALM model
    opt_params = [{'params': model.parameters(),  'lr': config['learning_rate']},          # Backbone
                  {'params': palm.protos,         'lr': config['learning_rate'] * 0.1}]  
    
    optimizer = optim.SGD(opt_params,
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, config, optimizer)
    palm.load_state(state['palm_path'])
    
    # Initialize dictionary for unknown labels
    unknown_labels = {}
    unknown_label_momentum = 0.9
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=True, dual_output=False)
            instance_dataset_val = Instance_Dataset(bags_val, [], transform=val_transform, warmup=True)
            train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'])
            val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], seed=1)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, collate_fn = collate_instance)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
            if state['warmup']:
                target_count = config['warmup_epochs']
            else:
                target_count = config['feature_extractor_train_count']
            
            
            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            
            for iteration in range(target_count): 
                model.train()
                losses = AverageMeter()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                train_pred = PredictionTracker()
                
                # Iterate over the training data
                for idx, (images, instance_labels, unique_ids) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)

                    # forward
                    optimizer.zero_grad()
                    _, _, instance_predictions, features = model(images, projector=True)
                    features.to(device)
                    
                    # Create masks for labeled and unlabeled data
                    labeled_mask = (instance_labels == 0) | (instance_labels == 1)

                    # Handle labeled instances (0 and 1)
                    if labeled_mask.any():
                        img = images[labeled_mask]
                        feat = features[labeled_mask]
                        labeled_instance_labels = instance_labels[labeled_mask]
                                        
                        # PALM Loss
                        palm_loss, loss_dict = palm(feat, labeled_instance_labels.long())
                    else:
                        palm_loss = torch.tensor(0.0).to(device)
                            
                    # Store raw predictions and targets
                    train_pred.update(instance_predictions, instance_labels, unique_ids)
                    
                    # Calculate BCE loss for confident instances
                    bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())

                    # Backward pass and optimization step
                    total_loss = palm_loss + bce_loss_value
                    total_loss.backward()
                    optimizer.step()
        
                    # Update the loss meter
                    losses.update(total_loss.item(), images[0].size(0))
                    
                    # Get predictions from PALM
                    with torch.no_grad():
                        # Calculate accuracy for PALM predictions
                        palm_predicted_classes, dist = palm.predict(features)
                        palm_correct = (palm_predicted_classes == instance_labels).sum().item()
                        palm_total_correct += palm_correct
                        
                        # Calculate accuracy for instance predictions
                        instance_predicted_classes = (instance_predictions) > 0.5
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += instance_labels.size(0)
                        
                    # Clean up
                    torch.cuda.empty_cache()

                # Calculate accuracies
                palm_train_acc = palm_total_correct / total_samples
                instance_train_acc = instance_total_correct / total_samples
                                
                
                
                # Validation loop
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
                        
                        # PALM Loss
                        palm_loss, _ = palm(features, instance_labels, update_prototypes=False)

                        # Calculate BCE loss
                        bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())

                        # Calculate total loss
                        total_loss = palm_loss + bce_loss_value
                        val_losses.update(total_loss.item(), images[0].size(0))

                        # Get predictions
                        palm_predicted_classes, dist = palm.predict(features)
                        instance_predicted_classes = (instance_predictions > 0.5)

                        # Calculate accuracy for PALM predictions
                        palm_correct = (palm_predicted_classes == instance_labels).sum().item()
                        palm_total_correct += palm_correct
                        
                        # Calculate accuracy for instance predictions
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += instance_labels.size(0)
                        
                        # Store raw predictions and targets
                        val_pred.update(instance_predictions, instance_labels, unique_id)
                        
                        # Clean up
                        torch.cuda.empty_cache()

                # Calculate accuracies
                palm_val_acc = palm_total_correct / total_samples if total_samples > 0 else 0
                instance_val_acc = instance_total_correct / total_samples if total_samples > 0 else 0

                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train Palm Acc: {palm_train_acc:.5f}, Train FC Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss: {val_losses.avg:.5f}, Val Palm Acc: {palm_val_acc:.5f}, Val FC Acc: {instance_val_acc:.5f}')
                
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    state['mode'] = 'instance'
                    save_metrics(config, state, train_pred, val_pred)
                    
                    if state['warmup']:
                        target_folder = state['head_folder']
                    else:
                        target_folder = state['model_folder']
                        
                    if state['warmup']:
                        save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, optimizer)
                        palm.save_state(os.path.join(target_folder, "palm_state.pkl"))
                        print("Saved checkpoint due to improved val_loss_instance")



        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            

        
            
        print('\nTraining Bag Aggregator')
        for iteration in range(config['MIL_train_count']):
            model.train()
            train_bag_logits = {}
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0
            train_pred = PredictionTracker()

            for (images, yb, instance_labels, unique_id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
                num_bags = len(images)
                optimizer.zero_grad()
                images, yb = images.cuda(), yb.cuda()

                # Forward pass
                bag_pred, _, instance_pred, features = model(images, pred_on=True)
                bag_pred = bag_pred.cuda()
                
                # Split the embeddings back into per-bag embeddings
                split_sizes = []
                for bag in images:
                    # Remove padded images (assuming padding is represented as zero tensors)
                    valid_images = bag[~(bag == 0).all(dim=1).all(dim=1).all(dim=1)] # Shape: [valid_images, 224, 224, 3]
                    split_sizes.append(valid_images.size(0))

                y_hat_per_bag = torch.split(instance_pred, split_sizes, dim=0)
                for i, y_h in enumerate(y_hat_per_bag):
                    train_bag_logits[unique_id[i].item()] = y_h.detach().cpu().numpy()
                
                
                bag_loss = BCE_loss(bag_pred, yb)
                bag_loss.backward()
                optimizer.step()
                
                total_loss += bag_loss.item() * yb.size(0)
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
                # Store raw predictions and targets
                train_pred.update(bag_pred, yb, unique_id)
                
                # Clean up
                torch.cuda.empty_cache()
                    
            
            
            train_loss = total_loss / total
            train_acc = correct / total
                    
                    
            # Evaluation phase
            model.eval()
            total = 0
            correct = 0
            total_val_loss = 0.0
            total_val_acc = 0.0
            val_pred = PredictionTracker()

            with torch.no_grad():
                for (images, yb, instance_labels, unique_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                    images, yb = images.cuda(), yb.cuda()
                    
                    # Forward pass
                    bag_pred, _, _, features = model(images, pred_on=True)
                    bag_pred = bag_pred.cuda()

                    # Calculate bag-level loss
                    loss = BCE_loss(bag_pred, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    predicted = (bag_pred > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()

                    # Store raw predictions and targets
                    val_pred.update(bag_pred, yb, unique_id)
                    
                    # Clean up
                    torch.cuda.empty_cache()
                        
            val_loss = total_val_loss / total
            val_acc = correct / total
            

            state['train_losses'].append(train_loss)
            state['valid_losses'].append(val_loss)    
            
            print(f"[{iteration+1}/{config['MIL_train_count']}] | Acc | Loss")
            print(f"Train | {train_acc:.4f} | {train_loss:.4f}")
            print(f"Val | {val_acc:.4f} | {val_loss:.4f}")

            # Save the model
            if val_loss < state['val_loss_bag']:
                state['val_loss_bag'] = val_loss
                state['mode'] = 'bag'
                
                if state['warmup']:
                    target_folder = state['head_folder']
                else:
                    target_folder = state['model_folder']

                
                save_state(state, config, train_acc, val_loss, val_acc, model, optimizer,)
                save_metrics(config, state, train_pred, val_pred)
                palm.save_state(os.path.join(target_folder, "palm_state.pkl"))
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1
                
                # Create selection mask
                state['selection_mask'] = create_momentum_predictions(train_bag_logits, state['selection_mask'])
                print("Created new sudo labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)


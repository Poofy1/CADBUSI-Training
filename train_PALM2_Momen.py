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
from archs.model_PALM2_solo import *
from data.bag_loader import *
from data.instance_loader import *
from loss.palm import PALM
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    


if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "PALM2_Momen_TEST_2"
    data_config = FishDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create bag datasets
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
    bag_dataset_val = BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False)
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=config['bag_batch_size'], collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_size=config['bag_batch_size'], collate_fn = collate_bag, drop_last=True)


    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    
    # LOSS INIT
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 90, lambda_pcon=3).cuda()
    BCE_loss = nn.BCELoss()
    CE_loss = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, optimizer, config)
    palm.load_state(state['palm_path'])
    
    # Initialize dictionary for unknown labels
    unknown_labels = {}
    unknown_label_momentum = 0.9
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=state['warmup'], dual_output=False)
            instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True)
            train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
            val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], strategy=1)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=4, collate_fn = collate_instance, pin_memory=True)
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
                
                # Iterate over the training data
                for idx, (images, instance_labels, unique_ids) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)

                    # forward
                    optimizer.zero_grad()
                    _, _, instance_predictions, features = model(images, projector=True)
                    features.to(device)
                    
                    # Create masks for labeled and unlabeled data
                    unlabeled_mask = (instance_labels == -1)
                    labeled_mask = ~unlabeled_mask  # This includes both 0 and 1 labels

                    # Handle labeled instances (0 and 1)
                    if labeled_mask.any():
                        img = images[labeled_mask]
                        feat = features[labeled_mask]
                        labeled_instance_labels = instance_labels[labeled_mask]
                                        
                        # PALM Loss
                        palm_loss, loss_dict = palm(feat, labeled_instance_labels)
                    else:
                        palm_loss = torch.tensor(0.0).to(device)

                    combined_labels = instance_labels.clone().float()
    
                    # Handle unlabeled instances (-1)
                    if unlabeled_mask.any():
                        unlabeled_features = features[unlabeled_mask]
                        unlabeled_indices = torch.where(unlabeled_mask)[0]
                        
                        with torch.no_grad():
                            proto_class, proto_dist = palm.predict(unlabeled_features)
                            
                            # Normalize distances
                            max_dist = torch.max(proto_dist)
                            min_dist = torch.min(proto_dist)
                            normalized_dist = (proto_dist - min_dist) / (max_dist - min_dist)
                        
                        for i, idx in enumerate(unlabeled_indices):
                            unique_id = unique_ids[idx]
                            
                            if unique_id not in unknown_labels:
                                unknown_labels[unique_id] = 0.5  # Initialize to 0.5 if not present
                            
                            # Calculate confidence from normalized distance
                            confidence = 1 - normalized_dist[i].item()

                            current_label = unknown_labels[unique_id]
                            new_label = proto_class[i].item()

                            # Adjust the momentum based on confidence
                            adjusted_momentum = unknown_label_momentum * (1 - confidence) + confidence

                            updated_label = adjusted_momentum * current_label + (1 - adjusted_momentum) * new_label
                            updated_label = max(0, min(1, updated_label))  # Clamp the updated label to [0, 1]
                            unknown_labels[unique_id] = updated_label
                            
                            # Update the combined_labels tensor with the new pseudo-label
                            combined_labels[idx] = updated_label
                    

                    # Calculate BCE loss for confident instances
                    bce_loss_value = BCE_loss(instance_predictions, combined_labels)

                    # Backward pass and optimization step
                    total_loss = palm_loss + bce_loss_value
                    total_loss.backward()
                    optimizer.step()
        
                    # Update the loss meter
                    losses.update(total_loss.item(), images[0].size(0))
                    
                    # Get predictions from PALM
                    with torch.no_grad():
                        # Calculate accuracy for PALM predictions
                        palm_predicted_classes, dist = palm.predict(features[labeled_mask])
                        palm_correct = (palm_predicted_classes == instance_labels[labeled_mask]).sum().item()
                        palm_total_correct += palm_correct
                        
                        # Calculate accuracy for instance predictions
                        instance_predicted_classes = (instance_predictions[labeled_mask]) > 0.5
                        instance_correct = (instance_predicted_classes == instance_labels[labeled_mask]).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += instance_labels[labeled_mask].size(0)

                # Calculate accuracies
                palm_train_acc = palm_total_correct / total_samples
                instance_train_acc = instance_total_correct / total_samples
                                
                
                
                # Validation loop
                model.eval()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                val_losses = AverageMeter()

                with torch.no_grad():
                    for idx, (images, instance_labels, _) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
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

                # Calculate accuracies
                palm_val_acc = palm_total_correct / total_samples if total_samples > 0 else 0
                instance_val_acc = instance_total_correct / total_samples if total_samples > 0 else 0

                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train Palm Acc: {palm_train_acc:.5f}, Train FC Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss: {val_losses.avg:.5f}, Val Palm Acc: {palm_val_acc:.5f}, Val FC Acc: {instance_val_acc:.5f}')
                
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    if state['warmup']:
                        target_folder = state['head_folder']
                        target_name = state['pretrained_name']
                    else:
                        target_folder = state['model_folder']
                        target_name = state['model_name']
                    all_targs = []
                    all_preds = []
                    
                    if state['warmup']:
                        save_state(state['epoch'], config['label_columns'], instance_train_acc, val_losses.avg, instance_val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
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
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0

            for (images, yb, instance_labels, id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
                num_bags = len(images)
                optimizer.zero_grad()

                # Forward pass
                bag_pred, _, instance_pred, features = model(images, pred_on=True)
                
                
                bag_loss = BCE_loss(bag_pred, yb)
                bag_loss.backward()
                optimizer.step()
                
                total_loss += bag_loss.item() * yb.size(0)
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                    
            
            
            train_loss = total_loss / total
            train_acc = correct / total
                    
                    
            # Evaluation phase
            model.eval()
            total = 0
            correct = 0
            total_val_loss = 0.0
            total_val_acc = 0.0
            all_targs = []
            all_preds = []

            with torch.no_grad():
                for (images, yb, instance_labels, id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 

                    # Forward pass
                    bag_pred, _, _, features = model(images, pred_on=True)

                    # Calculate bag-level loss
                    loss = BCE_loss(bag_pred, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    predicted = (bag_pred > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()

                    # Confusion Matrix data
                    all_targs.extend(yb.cpu().numpy())
                    if len(predicted.size()) == 0:
                        predicted = predicted.view(1)
                    all_preds.extend(predicted.cpu().detach().numpy())
                        
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
                if state['warmup']:
                    target_folder = state['head_folder']
                    target_name = state['pretrained_name']
                else:
                    target_folder = state['model_folder']
                    target_name = state['model_name']
                
                save_state(state['epoch'], config['label_columns'], train_acc, val_loss, val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                palm.save_state(os.path.join(target_folder, "palm_state.pkl"))
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1
                


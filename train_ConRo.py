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
from loss.ConRo import *
from util.eval_util import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import gc

if __name__ == '__main__':
    
    config = build_config()
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    model = build_model(config)    
    
    # LOSS INIT
    stage1_loss = ConRoLoss(temperature=0.7, alternating=True)
    stage2_loss = ConRoStage2Loss(temperature=0.7)
    BCE_loss = nn.BCEWithLogitsLoss()
    
    
    ops = {}
    ops['inst_optimizer'] = optim.Adam(model.parameters(),
                      lr=config['learning_rate'],
                      betas=(0.9, 0.999),
                      eps=1e-8,
                      weight_decay=0.001)
    """ops['inst_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)"""
    ops['bag_optimizer'] = optim.Adam(model.parameters(),
                      lr=config['learning_rate'],
                      betas=(0.9, 0.999),
                      eps=1e-8,
                      weight_decay=0.001)

    # MODEL INIT
    model, ops, state = setup_model(model, config, ops)
    scaler = GradScaler('cuda')
    state['warmup'] = False
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        if True:#not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataloader_train, instance_dataloader_val = get_instance_loaders(bags_train, bags_val, 
                                                                                      state, config, 
                                                                                      warmup=True)
            
            if state['warmup']:
                target_count = config['warmup_epochs']
            else:
                target_count = config['feature_extractor_train_count']
            
            
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            
            print("Computing global center of normal samples (v0)...")
            v0, radius = compute_global_v0(instance_dataloader_train, model)
            
            
            for iteration in range(target_count): 
                losses = AverageMeter()
                instance_total_correct = 0
                total_samples = 0
                train_pred = PredictionTracker()
                model.train()
                
                # Iterate over the training data
                for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
    
                    # forward
                    ops['inst_optimizer'].zero_grad()
                    with autocast('cuda'):
                        _, instance_predictions, features = model(images, projector=True)

                    # Calculate BCE loss
                    bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())
                    #bce_loss_value = 0
                    
                    
                    if state['warmup']:
                        # Calculate ConRo Stage 1 loss (alternating between contrastive and SVDD)
                        # Since we don't have auxiliary data, we pass None for auxiliary_embeddings and auxiliary_labels
                        conro_loss_value = stage1_loss(features, instance_labels, None, None)
                    else:
                        # Calculate ConRo Stage 2 loss
                        similar_malicious_embeddings, diverse_malicious_embeddings = generate_stage2_features(features, instance_labels, beta1=0.92, beta2=4.0, v0=v0)
                        conro_loss_value = stage2_loss(features, instance_labels, 
                                                        similar_malicious_embeddings=similar_malicious_embeddings, 
                                                        diverse_malicious_embeddings=diverse_malicious_embeddings,
                                                        auxiliary_embeddings=None, auxiliary_labels=None,)

                    # Backward pass and optimization step
                    total_loss = bce_loss_value + conro_loss_value
                    scaler.scale(total_loss).backward()
                    scaler.step(ops['inst_optimizer'])
                    scaler.update()
        
                    # Update the loss meter
                    losses.update(total_loss.item(), images[0].size(0))
                    
                    # Get predictions from PALM
                    instance_predictions = torch.sigmoid(instance_predictions)
                    with torch.no_grad():
                        instance_predicted_classes = (instance_predictions) > 0.5

                        # Calculate accuracy for instance predictions
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += instance_labels.size(0)
                        
                    # Store raw predictions and targets
                    train_pred.update(instance_predictions, instance_labels, unique_id)
                    
                    
                    # Clean up
                    torch.cuda.empty_cache()

                # Calculate accuracies
                instance_train_acc = instance_total_correct / total_samples
                                
                
                
                # Validation loop
                model.eval()
                instance_total_correct = 0
                total_samples = 0
                val_losses = AverageMeter()
                val_pred = PredictionTracker()

                with torch.no_grad():
                    for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        images = images.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)

                        # Forward pass
                        with autocast('cuda'):
                            _, instance_predictions, features = model(images, projector=True)
                        
                        # Get loss
                        bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())
                        #bce_loss_value = 0
                            
                        # Calculate ConRo Stage 1 loss (alternating between contrastive and SVDD)
                        # Since we don't have auxiliary data, we pass None for auxiliary_embeddings and auxiliary_labels
                        conro_loss_value = stage1_loss(features, instance_labels, None, None)
                        # No artificial features
                        
                        total_loss = bce_loss_value + conro_loss_value
                        val_losses.update(total_loss.item(), images[0].size(0))

                        # Get predictions
                        instance_predictions = torch.sigmoid(instance_predictions)
                        instance_predicted_classes = (instance_predictions) > 0.5
                        
                        # Calculate accuracy for instance predictions
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += instance_labels.size(0)
                        
                        # Store raw predictions and targets
                        val_pred.update(instance_predictions, instance_labels, unique_id)
                        
                        # Clean up
                        torch.cuda.empty_cache()

                # Calculate accuracies
                instance_val_acc = instance_total_correct / total_samples
                
                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train FC Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}, Val FC Acc: {instance_val_acc:.5f}')
                
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    state['mode'] = 'instance'
                    
                    if state['warmup']:
                        target_folder = state['head_folder']
                    else:
                        target_folder = state['model_folder']
                    
                    save_metrics(config, state, train_pred, val_pred)
                    
                    if True:#state['warmup']:
                        save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, ops)
                        print("Saved checkpoint due to improved val_loss_instance")





        """if state['pickup_warmup']: 
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
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for batch_idx, (images, yb, instance_labels, unique_id) in enumerate(tqdm(bag_dataloader_train, total=len(bag_dataloader_train))):
                num_bags = len(images)
                ops['bag_optimizer'].zero_grad()
                #images, yb = images.cuda(), yb.cuda()

                if not isinstance(images, list):
                    # If images is a padded tensor
                    images, yb = images.cuda(), yb.cuda()
                else:
                    # If images is a list of tensors
                    images = [img.cuda() for img in images]
                    yb = yb.cuda()
        
                # Forward pass
                with autocast('cuda'):
                    bag_pred, instance_pred, _ = model(images, pred_on=True)
                    bag_pred = bag_pred.cuda()
    
                # Split the embeddings back into per-bag embeddings
                split_sizes = []
                for bag in images:
                    # Remove padded images (assuming padding is represented as zero tensors)
                    valid_images = bag[~(bag == 0).all(dim=1).all(dim=1).all(dim=1)] # Shape: [valid_images, 224, 224, 3]
                    split_sizes.append(valid_images.size(0))

                #instance_pred = torch.cat(instance_pred, dim=0)
                y_hat_per_bag = torch.split(torch.sigmoid(instance_pred), split_sizes, dim=0)
                for i, y_h in enumerate(y_hat_per_bag):
                    train_bag_logits[unique_id[i].item()] = y_h.detach().cpu().numpy()
                
                bag_loss = BCE_loss(bag_pred, yb)
                scaler.scale(bag_loss).backward()
                scaler.step(ops['bag_optimizer'])
                scaler.update()
                
                bag_pred = torch.sigmoid(bag_pred)
                total_loss += bag_loss.item() * yb.size(0)
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
                # Store raw predictions and targets
                train_pred.update(bag_pred, yb, unique_id)
                
                
                ### It seems that GCP requires this cleanup?
                
                # Make sure we're explicitly cleaning up
                if isinstance(images, list):
                    for img in images:
                        img.detach()
                        del img
                else:
                    images.detach()
                    del images

                del instance_pred
                del y_hat_per_bag
                del bag_pred

                # Clean up
                torch.cuda.empty_cache()
                gc.collect()
                    
            
            
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
                    if not isinstance(images, list):
                        # If images is a padded tensor
                        images, yb = images.cuda(), yb.cuda()
                    else:
                        # If images is a list of tensors
                        images = [img.cuda() for img in images]
                        yb = yb.cuda()
                        
                        
                    # Forward pass
                    with autocast('cuda'):
                        bag_pred, _, features = model(images, pred_on=True)
                        bag_pred = bag_pred.cuda()

                    # Calculate bag-level loss
                    loss = BCE_loss(bag_pred, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    bag_pred = torch.sigmoid(bag_pred)
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

                
                save_state(state, config, train_acc, val_loss, val_acc, model, ops,)
                save_metrics(config, state, train_pred, val_pred)
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1
                
                # Create selection mask
                predictions_ratio = prediction_anchor_scheduler(state['epoch'], config)
                state['selection_mask'] = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)"""
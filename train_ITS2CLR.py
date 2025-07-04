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
from loss.SupCon import SupConLoss
from loss.SimCLR import SimCLRLoss
from loss.max_pooling import *
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
    supCon_pos = SupConLoss(pair_mode = 1)
    supCon_neg = SupConLoss(pair_mode = 2)
    BCE_loss = nn.BCEWithLogitsLoss()
    
    ops = {}
    ops['inst_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
    ops['bag_optimizer'] = optim.Adam(model.parameters(),
                        lr=config['learning_rate'],
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0.001)

    # MODEL INIT
    model, ops, state = setup_model(model, config, ops)
    scaler = GradScaler('cuda')
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            
            
            if state['warmup']:
                target_count = config['warmup_epochs']
                instance_dataloader_train, instance_dataloader_val = get_instance_loaders(bags_train, bags_val, 
                                                                                      state, config, 
                                                                                      use_bag_labels=True, dual_output = True)
            else:
                target_count = config['feature_extractor_train_count']
                instance_dataloader_train, _ = get_instance_loaders(bags_train, bags_val, 
                                                                                      state, config, 
                                                                                      only_pseudo=True, dual_output = True
                                                                                      #only_known=True, dual_output = True
                                                                                      )
                _, instance_dataloader_val = get_instance_loaders(bags_train, bags_val, 
                                                                                      state, config, 
                                                                                      #only_pseudo=True, dual_output = True
                                                                                      only_known=True, dual_output = True
                                                                                      )
                
            
            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            
            
            for iteration in range(target_count): 
                losses = AverageMeter()
                instance_total_correct = 0
                total_samples = 0
                train_pred = PredictionTracker()
                model.train()
                
                # Iterate over the training data
                for idx, (images, instance_labels, pseudo_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    bsz = instance_labels.shape[0]
                    im_q, im_k = images
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
                    images = [im_q, im_k]
                    images = torch.cat(images, dim=0).cuda()
                    
                    # forward
                    ops['inst_optimizer'].zero_grad()

                    with autocast('cuda'):
                        _, instance_predictions, features = model(images, projector=True)

                    zk, zq = torch.split(features, [bsz, bsz], dim=0)
                    features_for_supcon = torch.stack([zk, zq], dim=1)
                    
                    # Calculate loss
                    if state["warmup"]: # in warmup train on bag labels
                        loss_neg = supCon_neg(features_for_supcon, instance_labels.float(), instance_labels.float())
                        total_loss = loss_neg
                    else: # after warmup train on bag labels and pseudo labels
                        loss_neg = supCon_neg(features_for_supcon, pseudo_labels, instance_labels.float())
                        loss_pos = supCon_pos(features_for_supcon, pseudo_labels, instance_labels.float())
                        total_loss = loss_neg + loss_pos
                        
                    #total_loss = contrastive_loss(features_for_supcon, instance_labels.float())

                    # Backward pass and optimization step
                    scaler.scale(total_loss).backward()
                    scaler.step(ops['inst_optimizer'])
                    scaler.update()
        
                    # Update the loss meter
                    losses.update(total_loss.item(), bsz)
                        
                    # Store raw predictions and targets
                    #train_pred.update(instance_predictions, instance_labels, unique_id)
                    total_samples += instance_labels.size(0)
                    
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
                    for idx, (images, instance_labels, pseudo_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        bsz = instance_labels.shape[0]
                        im_q, im_k = images
                        im_q = im_q.cuda(non_blocking=True)
                        im_k = im_k.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)
                        images = [im_q, im_k]
                        images = torch.cat(images, dim=0).cuda()

                        # Forward pass
                        with autocast('cuda'):
                            _, instance_predictions, features = model(images, projector=True)
                        
                        zk, zq = torch.split(features, [bsz, bsz], dim=0)
                        features_for_supcon = torch.stack([zk, zq], dim=1)
                        
                        # Calculate loss
                        if state["warmup"]: # in warmup train on bag labels
                            loss_neg = supCon_neg(features_for_supcon, instance_labels.float(), instance_labels.float())
                            total_loss = loss_neg
                        else: # after warmup train on bag labels and pseudo labels
                            loss_neg = supCon_neg(features_for_supcon, pseudo_labels, instance_labels.float())
                            loss_pos = supCon_pos(features_for_supcon, pseudo_labels, instance_labels.float())
                            total_loss = loss_neg + loss_pos
                            
                        #total_loss = contrastive_loss(features_for_supcon, instance_labels.float())

                        # Store raw predictions and targets
                        #val_pred.update(instance_predictions, instance_labels, unique_id)
                        val_losses.update(total_loss.item(), bsz)
                        total_samples += instance_labels.size(0)
                        
                        # Clean up
                        torch.cuda.empty_cache()

                # Calculate accuracies
                instance_val_acc = instance_total_correct / total_samples
                
                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}')
                
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    state['mode'] = 'instance'
                    
                    if state['warmup']:
                        target_folder = state['head_folder']
                    else:
                        target_folder = state['model_folder']
                    
                    #save_metrics(config, state, train_pred, val_pred)
                    
                    if state['warmup']:
                        save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, ops)
                        print("Saved checkpoint due to improved val_loss_instance")





        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            

        
            
        print('\nTraining Bag Aggregator')
        
        # RESET MIL PARAMS
        model.reset_aggregator_parameters()
        
        for iteration in range(config['MIL_train_count']):
            model.train()
            bag_logits = {}
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0
            train_pred = PredictionTracker()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for batch_idx, (images, bag_labels, instance_labels, unique_id) in enumerate(tqdm(bag_dataloader_train, total=len(bag_dataloader_train))):
                num_bags = len(images)
                ops['bag_optimizer'].zero_grad()
                #images, bag_labels = images.cuda(), bag_labels.cuda()
                split_sizes = [bag.size(0) for bag in images]

                if not isinstance(images, list):
                    # If images is a padded tensor
                    images, bag_labels = images.cuda(), bag_labels.cuda()
                else:
                    # If images is a list of tensors
                    images = [img.cuda() for img in images]
                    bag_labels = bag_labels.cuda()
        
                # Forward pass
                with autocast('cuda'):
                    bag_pred, instance_predictions, _ = model(images, pred_on=True)
                    bag_pred = bag_pred.cuda()
    
                # Store both instance predictions and bag labels
                y_hat_per_bag = torch.split(torch.sigmoid(instance_predictions), split_sizes, dim=0)
                for i, y_h in enumerate(y_hat_per_bag):
                    bag_logits[unique_id[i].item()] = {
                        'instance_predictions': y_h.detach().cpu().numpy(),
                        'bag_label': bag_labels[i].detach().cpu().numpy()
                    }
                

                max_pool_loss = mil_max_loss(instance_predictions, bag_labels, split_sizes)
                bag_loss = BCE_loss(bag_pred, bag_labels)
                loss = bag_loss + max_pool_loss * .1
            
                scaler.scale(loss).backward()
                scaler.step(ops['bag_optimizer'])
                scaler.update()
                
                total_loss += loss.item() * bag_labels.size(0)
                predicted = (bag_pred > 0).float()
                total += bag_labels.size(0)
                correct += (predicted == bag_labels).sum().item()
                
                # Store raw predictions and targets
                train_pred.update(bag_pred, bag_labels, unique_id)
                
                
                ### It seems that GCP requires this cleanup?
                
                # Make sure we're explicitly cleaning up
                if isinstance(images, list):
                    for img in images:
                        img.detach()
                        del img
                else:
                    images.detach()
                    del images

                del instance_predictions
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
                for (images, bag_labels, instance_labels, unique_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                    if not isinstance(images, list):
                        # If images is a padded tensor
                        images, bag_labels = images.cuda(), bag_labels.cuda()
                    else:
                        # If images is a list of tensors
                        images = [img.cuda() for img in images]
                        bag_labels = bag_labels.cuda()
                    
                    split_sizes = [bag.size(0) for bag in images]
                        
                    # Forward pass
                    with autocast('cuda'):
                        bag_pred, instance_predictions, _ = model(images, pred_on=True)
                        bag_pred = bag_pred.cuda()

                    

                    predicted = (bag_pred > 0).float()
                    total += bag_labels.size(0)
                    correct += (predicted == bag_labels).sum().item()

                    # Store raw predictions and targets
                    val_pred.update(bag_pred, bag_labels, unique_id)
                    
                    y_hat_per_bag = torch.split(torch.sigmoid(instance_predictions), split_sizes, dim=0)
                    for i, y_h in enumerate(y_hat_per_bag):
                        bag_logits[unique_id[i].item()] = {
                            'instance_predictions': y_h.detach().cpu().numpy(),
                            'bag_label': bag_labels[i].detach().cpu().numpy()
                        }
                        
                    # Calculate bag-level loss
                    max_pool_loss = mil_max_loss(instance_predictions, bag_labels, split_sizes)
                    bag_loss = BCE_loss(bag_pred, bag_labels)
                    loss = bag_loss + max_pool_loss * .1
                    total_val_loss += loss.item() * bag_labels.size(0)
                    
                    # Clean up
                    torch.cuda.empty_cache()
                        
            val_loss = total_val_loss / total
            val_acc = correct / total
                
        
            state['train_losses'].append(train_loss)
            state['valid_losses'].append(val_loss)    
            state['epoch'] += 1
            
            print(f"Epoch {state['epoch'] + 1}")
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

                
                
                
                # Create selection mask
                predictions_ratio = prediction_anchor_scheduler(state['epoch'], config)
                state['selection_mask'] = create_selection_mask(bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)

import os
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.pseudo_labels import *
from data.bag_loader import *
from data.instance_loader import *
from util.eval_util import *
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        
        
if __name__ == '__main__':
    
    config = build_config()
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    model = build_model(config)    
    
    # LOSS INIT
    BCE_loss = nn.BCELoss()
    CE_crit = nn.CrossEntropyLoss()
    
    ops = {}
    ops['inst_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
    ops['bag_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)

    # MODEL INIT
    model, ops, state = setup_model(model, config, ops)

    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        torch.cuda.empty_cache()
        if not state['pickup_warmup']: # Are we resuming from a head model?
            
            
            instance_dataloader_train, instance_dataloader_val = get_instance_loaders(bags_train, bags_val, 
                                                                                      state, config, 
                                                                                      warmup=state['warmup'], dual_output=True)
            
            
            if state['warmup']:
                target_count = config['warmup_epochs']
            else:
                target_count = config['feature_extractor_train_count']
            
            
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')

            
            
            for iteration in range(target_count): 
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                model.train()
                
                train_iwscl_loss_total = AverageMeter()
                train_ce_loss_total = AverageMeter()
                train_pred = PredictionTracker()
                
                # Iterate over the training data
                for idx, ((im_q, im_k), instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)

                    # forward
                    ops['inst_optimizer'].zero_grad()
                    _, instance_predictions, _, feat_q, iwscl_loss, pseudo_labels = model(im_q, im_k, true_label = instance_labels, projector=True, bag_on=True)
                    feat_q.to(device)
                    
                    
                    # Calculate loss
                    ce_loss = CE_crit(instance_predictions, pseudo_labels.float())
                    total_loss = ce_loss + iwscl_loss
                    
                    # Backward pass and optimization step
                    total_loss.backward()
                    ops['inst_optimizer'].step()

                    # Update the loss meter
                    train_iwscl_loss_total.update(iwscl_loss.item(), instance_labels.size(0))
                    train_ce_loss_total.update(ce_loss.item(), instance_labels.size(0))
                    
                    # Get predictions
                    with torch.no_grad():
                        # Create mask for valid labels (0 and 1)
                        valid_mask = (instance_labels != -1)
                        
                        # Apply mask to get predictions and labels
                        valid_predictions = instance_predictions[valid_mask]
                        valid_labels = instance_labels[valid_mask]
                        
                        # Calculate accuracy for valid samples only
                        instance_predicted_classes = (valid_predictions > 0.5)
                        instance_correct = (instance_predicted_classes == valid_labels).sum().item()
                        instance_total_correct += instance_correct
                        total_samples += valid_mask.sum().item()
                    
                    # Store raw predictions and targets
                    train_pred.update(instance_predictions, instance_labels, unique_id)

                # Calculate accuracies
                palm_train_acc = palm_total_correct / total_samples
                instance_train_acc = instance_total_correct / total_samples

            
            
                # Validation loop
                model.eval()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                val_iwscl_loss_total = AverageMeter()
                val_ce_loss_total = AverageMeter()
                val_losses = AverageMeter()
                val_pred = PredictionTracker()

                with torch.no_grad():
                    for idx, ((im_q, im_k), instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        im_q = im_q.cuda(non_blocking=True)
                        im_k = im_k.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)

                        # Forward pass
                        _, instance_predictions, _, feat_q, iwscl_loss, pseudo_labels = model(im_q, true_label = instance_labels, projector=True, bag_on=True)
                        feat_q.to(device)
                        
                        # Calculate loss
                        ce_loss = CE_crit(instance_predictions, pseudo_labels.float())
                        total_loss = ce_loss + iwscl_loss
                        
                        val_iwscl_loss_total.update(iwscl_loss.item(), instance_labels.size(0))
                        val_ce_loss_total.update(ce_loss.item(), instance_labels.size(0))
                        val_losses.update(total_loss.item(), instance_labels.size(0))

                        # Get predictions
                        instance_predicted_classes = (instance_predictions) > 0.5
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        total_samples += instance_labels.size(0)
                        
                        # Store raw predictions and targets
                        val_pred.update(instance_predictions, instance_labels, unique_id)
                    
                    

                # Calculate accuracies
                palm_val_acc = palm_total_correct / total_samples
                instance_val_acc = instance_total_correct / total_samples
                
                
                print(f'[{iteration+1}/{target_count}] Train | iwscl Loss: {train_iwscl_loss_total.avg:.5f}, CE Loss: {train_ce_loss_total.avg:.5f}, Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val   | iwscl Loss: {val_iwscl_loss_total.avg:.5f}, CE Loss: {val_ce_loss_total.avg:.5f}, Acc: {instance_val_acc:.5f}')
                
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    state['mode'] = 'instance'
                    save_metrics(config, state, train_pred, val_pred)
                    
                    if state['warmup']:
                        save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, ops)
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
                ops['bag_optimizer'].zero_grad()

                # Forward pass
                bag_pred, instance_pred, _, _, _, _ = model(images, bag_on=True)
                
                bag_pred = torch.clamp(bag_pred, min=0.000001, max=.999999)

                bag_loss = BCE_loss(bag_pred, yb)
                bag_loss.backward()
                ops['bag_optimizer'].step()
                
                total_loss += bag_loss.item() * yb.size(0)
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
                # Store raw predictions and targets
                train_pred.update(bag_pred, yb, unique_id)
                    
            
            
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

                    # Forward pass
                    bag_pred, _, _, _, _, _ = model(images, bag_on=True, val_on = True)
                    bag_pred = torch.clamp(bag_pred, min=0.000001, max=.999999)
                    # Calculate bag-level loss
                    loss = BCE_loss(bag_pred, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    predicted = (bag_pred > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()

                    # Store raw predictions and targets
                    val_pred.update(bag_pred, yb, unique_id)
                        
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

                save_state(state, config, train_acc, val_loss, val_acc, model, ops,)
                save_metrics(config, state, train_pred, val_pred)
                print("Saved checkpoint due to improved val_loss_bag")
                
                state['epoch'] += 1


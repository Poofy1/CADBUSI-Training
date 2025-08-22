import os
import torch
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
import torch.optim as optim

from data.save_arch import *
from util.Gen_ITS2CLR_util import *
from data.format_data import *
from data.pseudo_labels import *
from data.bag_loader import *
from data.instance_loader import *
from loss.max_pooling import *
from util.eval_util import *
from config import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.benchmark = True

import os
"""os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Instead of fixing device to cuda:0, just use 'cuda' if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optional: This is often useful for debugging but can degrade performance on multi-GPU setups
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(torch.cuda.device_count(), "GPUs detected.")
"""
if __name__ == '__main__':
    config = build_config()
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    model = build_model(config)

    """# Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)"""

    loss_func = nn.BCEWithLogitsLoss()

    ops = {}
    ops['bag_optimizer'] = optim.Adam(model.parameters(),
                        lr=config['learning_rate'],
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0.001)


    # MODEL INIT
    model, ops, state = setup_model(model, config, ops)
    
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    
    print("Training Data...")
    # Training loop
    while state['epoch'] < config['total_epochs']:
        # Training phase
        model.train()
        total_loss = 0.0
        total_acc = 0
        total = 0
        correct = [0] * config['num_labels']
        bag_logits = {}
        train_pred = PredictionTracker()
                
                
        for (all_images, bag_labels, instance_labels, bag_ids) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)): 
            bag_labels = bag_labels.cuda()
            all_images = [img.cuda() for img in all_images]
            split_sizes = [bag.size(0) for bag in all_images]
            
            ops['bag_optimizer'].zero_grad()
            
            physical_delta_x = extract_float_input(instance_labels, 'PhysicalDeltaX')
            bag_pred, instance_predictions, _ = model(all_images, float_input=physical_delta_x, pred_on=True)
            
            max_pool_loss = mil_max_loss(instance_predictions, bag_labels, split_sizes)
            bag_loss = loss_func(bag_pred, bag_labels)
            loss = bag_loss + max_pool_loss * .1
            
            loss.backward()
            ops['bag_optimizer'].step()

            total_loss += loss.item() * len(all_images)
            predicted = (bag_pred > 0).float()
            total += len(bag_labels)
            

            # Store both instance predictions and bag labels
            y_hat_per_bag = torch.split(torch.sigmoid(instance_predictions), split_sizes, dim=0)
            for i, y_h in enumerate(y_hat_per_bag):
                bag_logits[bag_ids[i].item()] = {
                    'instance_predictions': y_h.detach().cpu().numpy(),
                    'bag_label': bag_labels[i].detach().cpu().numpy()
                }
                    
            for label_idx in range(config['num_labels']):
                correct[label_idx] += (predicted[:, label_idx] == bag_labels[:, label_idx]).sum().item()
                
            # Store raw predictions and targets
            train_pred.update(bag_pred, bag_labels, bag_ids)
            
        train_loss = total_loss / total
        train_acc = [total_correct / total for total_correct in correct]


        # Evaluation phase
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        total = 0
        correct = [0] * config['num_labels']
        val_pred = PredictionTracker()
            
        with torch.no_grad():
            for (all_images, bag_labels, instance_labels, bag_ids) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                bag_labels = bag_labels.cuda()
                all_images = [img.cuda() for img in all_images]
                split_sizes = [bag.size(0) for bag in all_images]
                
                physical_delta_x = extract_float_input(instance_labels, 'PhysicalDeltaX')
                bag_pred, instance_predictions, _ = model(all_images, float_input=physical_delta_x, pred_on=True)
                
                max_pool_loss = mil_max_loss(instance_predictions, bag_labels, split_sizes)
                bag_loss = loss_func(bag_pred, bag_labels)
                loss = bag_loss + max_pool_loss * .1
                
                total_val_loss += loss.item() * len(all_images)
                predicted = (bag_pred > 0).float()
                total += len(bag_labels)
                
                # Store both instance predictions and bag labels
                y_hat_per_bag = torch.split(torch.sigmoid(instance_predictions), split_sizes, dim=0)
                for i, y_h in enumerate(y_hat_per_bag):
                    bag_logits[bag_ids[i].item()] = {
                        'instance_predictions': y_h.detach().cpu().numpy(),
                        'bag_label': bag_labels[i].detach().cpu().numpy()
                    }
                            
                for label_idx in range(config['num_labels']):
                    correct[label_idx] += (predicted[:, label_idx] == bag_labels[:, label_idx]).sum().item()
                
                # Store raw predictions and targets
                val_pred.update(bag_pred, bag_labels, bag_ids)

        val_loss = total_val_loss / total
        val_acc = [total_correct / total for total_correct in correct]
        
        train_losses_over_epochs.append(train_loss)
        valid_losses_over_epochs.append(val_loss)
        
        # Constructing header with label names
        acc_headers = " | ".join(f"Acc ({name})" for name in config['label_columns'])
        header = f"Epoch {state['epoch']+1} | {acc_headers} | Loss"

        # Constructing training and validation accuracy strings
        train_acc_str = " | ".join(f"{acc:.4f}" for acc in train_acc)
        val_acc_str = " | ".join(f"{acc:.4f}" for acc in val_acc)

        # Printing epoch summary
        print(header)
        print(f"Train   | {train_acc_str} | {train_loss:.4f}")
        print(f"Val     | {val_acc_str} | {val_loss:.4f}")
        
        target_folder = state['head_folder']
        target_name = state['pretrained_name']
        state['epoch'] += 1
        
        # Save the model
        if val_loss < state['val_loss_bag']:
            state['val_loss_bag'] = val_loss  # Update the best validation accuracy
            state['mode'] = 'bag'
            
            # Create selection mask
            #predictions_ratio = prediction_anchor_scheduler(state['epoch'], config)
            state['selection_mask'] = create_selection_mask(bag_logits, 1.0)
            
            # Save selection
            with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                pickle.dump(state['selection_mask'], file)

            save_state(state, config, train_acc, val_loss, val_acc, model, ops,)
            save_metrics(config, state, train_pred, val_pred)
            print("Saved checkpoint due to improved val_loss")
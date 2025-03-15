import os
import torch
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
import torch.optim as optim

from data.save_arch import *
from util.Gen_ITS2CLR_util import *
from data.format_data import *
from data.sudo_labels import *
from data.bag_loader import *
from data.instance_loader import *
from loss.FocalLoss import *
from util.eval_util import *
from config import *

torch.backends.cudnn.benchmark = True

import os
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Instead of fixing device to cuda:0, just use 'cuda' if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optional: This is often useful for debugging but can degrade performance on multi-GPU setups
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == '__main__':
    print(torch.cuda.device_count(), "GPUs detected.")
    print(f'device: {device}')
    # Config
    model_version = '1'
    head_name = "TEST500"
    data_config = LesionDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = build_model(config)

    # Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    loss_func = nn.BCELoss()

    ops = {}
    ops['bag_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
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
        correct = [0] * num_labels
        train_pred = PredictionTracker()
                
                
        for (all_images, bag_labels, instance_labels, bag_ids) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)): 
            bag_labels = bag_labels.cuda()
            ops['bag_optimizer'].zero_grad()
            
            bag_pred, _, _, _ = model(all_images.cuda())

            loss = loss_func(bag_pred, bag_labels)

            loss.backward()
            ops['bag_optimizer'].step()

            total_loss += loss.item() * len(all_images)
            predicted = (bag_pred > .5).float()
            total += len(bag_labels)
            
            for label_idx in range(num_labels):
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
        correct = [0] * num_labels
        val_pred = PredictionTracker()
            
        with torch.no_grad():
            for (all_images, bag_labels, instance_labels, bag_ids) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                bag_labels = bag_labels.cuda()
                bag_pred, _, _, _ = model(all_images.cuda())

                loss = loss_func(bag_pred, bag_labels)
                
                total_val_loss += loss.item() * len(all_images)
                predicted = (bag_pred > .5).float()
                total += len(bag_labels)
                
                for label_idx in range(num_labels):
                    correct[label_idx] += (predicted[:, label_idx] == bag_labels[:, label_idx]).sum().item()
                
                # Store raw predictions and targets
                val_pred.update(bag_pred, yb, bag_ids)

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
        
        # Save the model
        if val_loss < state['val_loss_bag']:
            state['val_loss_bag'] = val_loss  # Update the best validation accuracy
            state['mode'] = 'bag'

            save_state(state, config, train_acc, val_loss, val_acc, model, ops,)
            save_metrics(config, state, train_pred, val_pred)
            print("Saved checkpoint due to improved val_loss")
import os
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from torch.optim import Adam
import torch.optim as optim
from data.format_data import *
from data.bag_loader import *
from util.eval_util import *
from config import *
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "ABMIL_OFFICAL"
    data_config = FishDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])
    
    # Create Model
    model = build_model(config)
        
        
    loss_func = nn.BCELoss()
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    
    ops = {}
    ops['bag_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)

    # MODEL INIT
    model, ops, state = setup_model(model, config, ops)
    
    
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
                
                
        for (data, yb, instance_yb, unique_id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)): 
            xb, yb = data, yb.cuda()
            
            ops['bag_optimizer'].zero_grad()
            
            bag_pred, _, _, _ = model(xb)

            loss = loss_func(bag_pred, yb)

            loss.backward()
            ops['bag_optimizer'].step()

            total_loss += loss.item() * len(xb)
            predicted = (bag_pred > .5).float()
            total += yb.size(0)
            
            for label_idx in range(num_labels):
                correct[label_idx] += (predicted[:, label_idx] == yb[:, label_idx]).sum().item()
                
            # Store raw predictions and targets
            train_pred.update(bag_pred, yb, unique_id)
            
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
            for (data, yb, instance_yb, unique_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                xb, yb = data, yb.cuda()

                bag_pred, _, _, _ = model(xb)

                
                loss = loss_func(bag_pred, yb)
                
                total_val_loss += loss.item() * len(xb)
                predicted = (bag_pred > .5).float()
                total += yb.size(0)
                
                for label_idx in range(num_labels):
                    correct[label_idx] += (predicted[:, label_idx] == yb[:, label_idx]).sum().item()
                
                # Store raw predictions and targets
                val_pred.update(bag_pred, yb, unique_id)

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
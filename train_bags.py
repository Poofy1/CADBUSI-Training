import os
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
import gc
from data.format_data import *
from data.sudo_labels import *
from archs.model_GenSCL import *
from data.bag_loader import *
from data.instance_loader import *
from loss.FocalLoss import *
from loss.contrastive import *
from util.eval_util import *
from config import *
from torch.cuda import memory_summary
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class BCELossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Apply label smoothing
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        # Calculate BCE
        bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))
        return bce.mean()


def calculate_instance_accuracy(attention_preds, instance_labels_list):
    correct = 0
    total = 0
    
    for bag_idx, bag_labels in enumerate(instance_labels_list):
        # Convert list of single-item tensors to a single tensor
        bag_label_values = torch.tensor([label.item() for label in bag_labels]).cuda()
        
        # Get predictions for current bag
        bag_pred = attention_preds[bag_idx]
        
        # Create mask for valid labels (not equal to -1)
        valid_mask = bag_label_values != -1
        
        # Filter out -1 labels and corresponding predictions
        valid_labels = bag_label_values[valid_mask]
        valid_preds = bag_pred[valid_mask]
        
        # Convert to binary predictions
        valid_preds = (valid_preds > 0.5).float()
        
        # Count correct predictions
        correct += (valid_preds == valid_labels).sum().item()
        total += len(valid_labels)
    
    return correct / total if total > 0 else 0

def get_prediction_stats(predictions):
    pred_numpy = predictions.detach().cpu().numpy()
    pred_binary = (pred_numpy > 0.5).astype(float)
    zeros = (pred_binary == 0).sum()
    ones = (pred_binary == 1).sum()
    total = zeros + ones
    return (zeros/total * 100), (ones/total * 100)

    
if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "TEST46"
    data_config = FishDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, _, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])
    
    # Procedural Bags
    bag_dataset_train = SyntheticBagDataset(bags_train, transform=train_transform, min_bag_size=config['min_bag_size'], max_bag_size=config['max_bag_size'])
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=config['bag_batch_size'], collate_fn=collate_bag)
    

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    


    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001

    #BCE_loss = BCELossWithSmoothing(smoothing=config['label_smoothing']).cuda()
    BCE_loss = nn.BCELoss()
    #BCE_loss = FocalLoss()
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, config, optimizer)
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        
        
        print('\nTraining Bag Aggregator')
        model.train()
        total_loss = 0.0
        total_acc = 0
        total = 0
        correct = 0
        running_loss = 0
        train_pred = PredictionTracker()
        
        train_bag_zeros = 0
        train_bag_ones = 0
        train_attn_zeros = 0
        train_attn_ones = 0
        n_train_batches = len(bag_dataloader_train)
        
        train_instance_acc = 0
        
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)

        for (images, yb, instance_labels, unique_id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
            num_bags = len(images)
            optimizer.zero_grad()
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            bag_pred, attention_pred, instance_pred, features = model(images, pred_on=True, projector=True)
        
            
            # Calculate bag loss
            bag_loss = BCE_loss(bag_pred, yb)
            
            # Track bag predictions
            bag_zeros, bag_ones = get_prediction_stats(bag_pred)
            train_bag_zeros += bag_zeros
            train_bag_ones += bag_ones
            
            # Track attention predictions
            attention_pred_cat = torch.cat([pred for pred in attention_pred])
            attn_zeros, attn_ones = get_prediction_stats(attention_pred_cat)
            train_attn_zeros += attn_zeros
            train_attn_ones += attn_ones

            # Calculate instance accuracy
            train_batch_instance_acc = calculate_instance_accuracy(attention_pred, instance_labels)
            train_instance_acc += train_batch_instance_acc
                            
            total_loss = bag_loss

            total_loss.backward()
            optimizer.step()
            
            # Track bag-level metrics
            batch_size = yb.size(0)
            running_loss += total_loss.item() * batch_size
            predicted = (bag_pred > 0.5).float()
            total += batch_size
            correct += (predicted == yb).sum().item()
            
            # Store raw predictions and targets
            train_pred.update(bag_pred, yb, unique_id)
            
            torch.cuda.empty_cache()

        # Calculate final metrics
        train_loss = running_loss / total
        train_acc = correct / total

        train_instance_acc /= n_train_batches
        
        # Average over batches
        train_bag_zeros /= n_train_batches
        train_bag_ones /= n_train_batches
        train_attn_zeros /= n_train_batches
        train_attn_ones /= n_train_batches
                
        # Evaluation phase
        model.eval()
        total = 0
        correct = 0
        total_val_loss = 0.0
        val_pred = PredictionTracker()
        
        val_instance_acc = 0
        n_val_batches = len(bag_dataloader_val)

        with torch.no_grad():
            for (images, yb, instance_labels, unique_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                # Forward pass
                bag_pred, attention_pred, instance_pred, features = model(images, projector=True, pred_on=True)

                
                # Calculate bag-level loss
                bag_loss = BCE_loss(bag_pred, yb)
                    
                total_loss = bag_loss

                total_val_loss += total_loss.item() * yb.size(0)

                # Bag-level metrics
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()

                # Store raw predictions and targets
                val_pred.update(bag_pred, yb, unique_id)
                
                
                # Handle instance labels properly
                val_batch_instance_acc = calculate_instance_accuracy(attention_pred, instance_labels)
                val_instance_acc += val_batch_instance_acc
                
                torch.cuda.empty_cache()
                            
        val_loss = total_val_loss / total
        val_acc = correct / total
        
        val_instance_acc /= n_val_batches

        state['train_losses'].append(train_loss)
        state['valid_losses'].append(val_loss)    

        print(f"\n[Epoch {state['epoch']+1}/{config['total_epochs']}]")
        print("-" * 70)
        print(f"{'':15} {'Bag Acc':>10} {'Inst Acc':>10} {'Loss':>10} {'Bag 0%':>8} {'Bag 1%':>8} {'Attn 0%':>8} {'Attn 1%':>8}")
        print("-" * 70)
        print(f"{'Train':15} {train_acc:>10.4f} {train_instance_acc:>10.4f} {train_loss:>8.4f} {train_bag_zeros:>8.1f} {train_bag_ones:>8.1f} {train_attn_zeros:>8.1f} {train_attn_ones:>8.1f}")
        print(f"{'Validation':15} {val_acc:>10.4f} {val_instance_acc:>10.4f} {val_loss:>10.4f}")
        print("-" * 70)
                    
    
        #print(memory_summary(device=None, abbreviated=False))
        state['epoch'] += 1
        
        # Save the model
        if val_loss < state['val_loss_instance']:
            state['val_loss_instance'] = val_loss
            state['mode'] = 'bag'
            
            save_state(state, config, train_acc, val_loss, val_acc, model, optimizer)
            save_metrics(config, state, train_pred, val_pred)
            print("Saved checkpoint due to improved val_loss_instance")


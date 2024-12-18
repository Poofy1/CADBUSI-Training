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





    
if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "TEST"
    data_config = FishDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    
    # LOSS INIT
    config.update({
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'label_smoothing': 0.1,
        'warmup_epochs': 5,
        'cosine_schedule': True
    })

    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001

    #BCE_loss = BCELossWithSmoothing(smoothing=config['label_smoothing']).cuda()
    BCE_loss = nn.BCELoss()
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, config, optimizer)
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        print('\nTraining Bag Aggregator')
        for iteration in range(10):
            model.train()
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0
            attention_correct = 0
            attention_total = 0
            running_loss = 0
            instance_correct = 0
            instance_total = 0
            train_pred = PredictionTracker()
            
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
                
                # Convert the nested structure to a proper tensor
                instance_labels = [float(tensor.item()) for sublist in instance_labels for tensor in sublist]
                attention_pred = torch.cat([pred for pred in attention_pred])
                instance_labels = torch.tensor(instance_labels).cuda()
                valid_mask = instance_labels != -1
                
                # Calculate instance-level losses only for valid labels
                if valid_mask.sum() > 0:  # Only if we have valid instance labels
                    # Get predictions and labels for valid instances
                    valid_attention_pred = attention_pred[valid_mask].cuda()
                    #valid_instance_pred = instance_pred[valid_mask].cuda()
                    valid_instance_labels = instance_labels[valid_mask]

                    # Calculate contrastive loss for valid features
                    valid_features = features[valid_mask]
                    contrastive_loss_value = contrastive_loss(valid_features, valid_instance_labels)
        
                    
                    attention_loss = BCE_loss(valid_attention_pred, valid_instance_labels)
                    instance_loss = 0 #BCE_loss(valid_instance_pred, valid_instance_labels)
                    
                    # Combine losses with weights
                    λ1, λ2 = 1, 1  # Adjust these weights as needed
                    total_loss = bag_loss + λ1 * attention_loss + λ2 * instance_loss + contrastive_loss_value 
                else:
                    total_loss = bag_loss

                total_loss.backward()
                optimizer.step()
                
                # Track bag-level metrics
                batch_size = yb.size(0)
                running_loss += total_loss.item() * batch_size
                predicted = (bag_pred > 0.5).float()
                total += batch_size
                correct += (predicted == yb).sum().item()
                
                # Track instance-level metrics for valid instances
                if valid_mask.sum() > 0:
                    # Attention prediction metrics
                    attention_predicted = (valid_attention_pred > 0.5).float()
                    attention_correct += (attention_predicted == valid_instance_labels).sum().item()
                    attention_total += valid_mask.sum().item()
                    
                    # Instance prediction metrics
                    """instance_predicted = (valid_instance_pred > 0.5).float()
                    instance_correct += (instance_predicted == valid_instance_labels).sum().item()
                    instance_total += valid_mask.sum().item()"""
                    
                    instance_correct = 0
                    instance_total = 0
                
                # Store raw predictions and targets
                train_pred.update(bag_pred, yb, unique_id)

            # Calculate final metrics
            train_loss = running_loss / total
            train_acc = correct / total

            # Calculate instance-level accuracies
            attention_acc = attention_correct / attention_total if attention_total > 0 else 0
            instance_acc = instance_correct / instance_total if instance_total > 0 else 0
                    
                    
            # Evaluation phase
            model.eval()
            total = 0
            correct = 0
            total_val_loss = 0.0
            val_attention_correct = 0
            val_attention_total = 0
            val_instance_correct = 0
            val_instance_total = 0
            val_pred = PredictionTracker()

            with torch.no_grad():
                for (images, yb, instance_labels, unique_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                    # Forward pass
                    bag_pred, attention_pred, instance_pred, features = model(images, projector=True, pred_on=True)

                    # Calculate bag-level loss
                    bag_loss = BCE_loss(bag_pred, yb)
                    
                    # Convert the nested structure to a proper tensor
                    instance_labels = [float(tensor.item()) for sublist in instance_labels for tensor in sublist]
                    attention_pred = torch.cat([pred for pred in attention_pred])
                    instance_labels = torch.tensor(instance_labels).cuda()
                    valid_mask = instance_labels != -1
                    
                    # Calculate instance-level losses only for valid labels
                    if valid_mask.sum() > 0:
                        valid_attention_pred = attention_pred[valid_mask].cuda()
                        #valid_instance_pred = instance_pred[valid_mask].cuda()
                        valid_instance_labels = instance_labels[valid_mask]
                        
                        attention_loss = BCE_loss(valid_attention_pred, valid_instance_labels)
                        instance_loss = 0 #BCE_loss(valid_instance_pred, valid_instance_labels)
                        
                        # Calculate contrastive loss for valid features
                        valid_features = features[valid_mask]
                        contrastive_loss_value = contrastive_loss(valid_features, valid_instance_labels)
                        
                        λ1, λ2 = 1, 1
                        total_loss = bag_loss + λ1 * attention_loss + λ2 * instance_loss + contrastive_loss_value
                    else:
                        total_loss = bag_loss

                    total_val_loss += total_loss.item() * yb.size(0)

                    # Bag-level metrics
                    predicted = (bag_pred > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()

                    # Instance-level metrics for valid instances
                    if valid_mask.sum() > 0:
                        # Attention prediction metrics
                        val_attention_predicted = (valid_attention_pred > 0.5).float()
                        val_attention_correct += (val_attention_predicted == valid_instance_labels).sum().item()
                        val_attention_total += valid_mask.sum().item()
                        
                        # Instance prediction metrics
                        """val_instance_predicted = (valid_instance_pred > 0.5).float()
                        val_instance_correct += (val_instance_predicted == valid_instance_labels).sum().item()
                        val_instance_total += valid_mask.sum().item()"""
                        
                        val_instance_correct = 0
                        val_instance_total = 0

                    # Store raw predictions and targets
                    val_pred.update(bag_pred, yb, unique_id)
                                
            val_loss = total_val_loss / total
            val_acc = correct / total
            val_attention_acc = val_attention_correct / val_attention_total if val_attention_total > 0 else 0
            val_instance_acc = val_instance_correct / val_instance_total if val_instance_total > 0 else 0

            state['train_losses'].append(train_loss)
            state['valid_losses'].append(val_loss)    

            print(f"\n[Epoch {iteration+1}/{config['MIL_train_count']}]")
            print("-" * 50)
            print(f"{'':15} {'Bag Acc':>10} {'Attn Acc':>10} {'Inst Acc':>10} {'Loss':>10}")
            print("-" * 50)
            print(f"{'Train':15} {train_acc:>10.4f} {attention_acc:>10.4f} {instance_acc:>10.4f} {train_loss:>10.4f}")
            print(f"{'Validation':15} {val_acc:>10.4f} {val_attention_acc:>10.4f} {val_instance_acc:>10.4f} {val_loss:>10.4f}")
            print("-" * 50)
                        
        
            #print(memory_summary(device=None, abbreviated=False))
            
            # Save the model
            if val_loss < state['val_loss_instance']:
                state['val_loss_instance'] = val_loss
                state['mode'] = 'instance'
                
                save_state(state, config, train_acc, val_loss, val_acc, model, optimizer)
                save_metrics(config, state, train_pred, val_pred)
                print("Saved checkpoint due to improved val_loss_instance")


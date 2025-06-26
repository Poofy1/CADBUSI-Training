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
from config import *
from util.eval_util import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def apply_mixup(images, labels, alpha=1.0, device='cuda'):
    """
    Apply mixup augmentation to images and labels.
    
    Args:
        images (torch.Tensor): Input images
        labels (torch.Tensor): Input labels
        alpha (float): Mixup interpolation strength
        device (str): Device to use for computation
        
    Returns:
        tuple: (mixed_images, labels_a, labels_b, lam)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_images = lam * images + (1 - lam) * images[index, :]
    labels_a, labels_b = labels, labels[index]
    
    return mixed_images, labels_a, labels_b, lam


if __name__ == '__main__':
    
    config = build_config()
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    use_mixup = False
    model = build_model(config)     
    
    # LOSS INIT
    BCE_loss = nn.BCEWithLogitsLoss()

    
    ops = {}
    ops['inst_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)

    # MODEL INIT
    model, ops, state = setup_model(model, config, ops)
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        instance_dataloader_train, instance_dataloader_val = get_instance_loaders(bags_train, bags_val, 
                                                                                      state, config, 
                                                                                      only_known=True)

        
        if state['warmup']:
            target_count = config['warmup_epochs']
        else:
            target_count = config['feature_extractor_train_count']
        

        print('Training Feature Extractor')
        print(f'Warmup Mode: {state["warmup"]}')
        
        alpha = 1.0
        
        for iteration in range(target_count): 
            losses = AverageMeter()
            instance_total_correct = 0
            total_samples = 0
            model.train()
            train_pred = PredictionTracker()
            
            # Iterate over the training data
            for idx, (images, instance_labels, pseudo_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                images = images.cuda(non_blocking=True)
                instance_labels = instance_labels.cuda(non_blocking=True)

                if use_mixup:
                    # Apply mixup augmentation
                    mixed_images, labels_a, labels_b, lam = apply_mixup(
                        images, 
                        instance_labels, 
                        alpha=1, 
                        device=device
                    )
                    
                    # Forward pass with mixed images
                    ops['inst_optimizer'].zero_grad()
                    _, instance_predictions, features = model(mixed_images, projector=True)
                    features = features.to(device)

                    # Calculate mixed BCE loss
                    loss_a = BCE_loss(instance_predictions, labels_a.float())
                    loss_b = BCE_loss(instance_predictions, labels_b.float())
                    bce_loss_value = lam * loss_a + (1 - lam) * loss_b
                    
                    # Calculate accuracy considering both original and mixed labels
                    predicted = (instance_predictions > 0).float()
                    correct_a = (predicted == labels_a).float()
                    correct_b = (predicted == labels_b).float()
                    correct = lam * correct_a + (1 - lam) * correct_b
                    
                else:
                    # Standard forward pass without mixup
                    ops['inst_optimizer'].zero_grad()
                    _, instance_predictions, features = model(images, projector=True)
                    features = features.to(device)

                    # Standard BCE loss
                    bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())
                    
                    # Standard accuracy calculation
                    predicted = (instance_predictions > 0).float()
                    correct = (predicted == instance_labels).float()
                
                # Common steps for both cases
                total_loss = bce_loss_value
                total_loss.backward()
                ops['inst_optimizer'].step()

                # Update metrics
                losses.update(total_loss.item(), images.size(0))
                instance_total_correct += correct.sum().item()
                total_samples += instance_labels.numel()
                
                # Store raw predictions and targets
                train_pred.update(instance_predictions, instance_labels, unique_id)

            # Calculate training accuracy
            instance_train_acc = instance_total_correct / total_samples
            

            # Validation loop
            model.eval()
            instance_total_correct = 0
            total_samples = 0
            val_losses = AverageMeter()
            val_pred = PredictionTracker()

            with torch.no_grad():
                for idx, (images, instance_labels, pseudo_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)

                    # Forward pass
                    _, instance_predictions, features = model(images, projector=True)
                    features.to(device)
                    
                    # Get loss
                    bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())
                    total_loss = bce_loss_value
                    val_losses.update(total_loss.item(), images[0].size(0))
                    
                    # Calculate correct predictions
                    predicted = (instance_predictions > 0).float()
                    correct = (predicted == instance_labels).float()
                    instance_total_correct += correct.sum().item()
                    total_samples += instance_labels.numel()
                    
                    # Store raw predictions and targets
                    val_pred.update(instance_predictions, instance_labels, unique_id)

            # Calculate validation accuracy
            instance_val_acc = instance_total_correct / total_samples
            
            print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train FC Acc: {instance_train_acc:.5f}')
            print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}, Val FC Acc: {instance_val_acc:.5f}')
            
            
                    
            # Save the model
            if val_losses.avg < state['val_loss_instance']:
                state['val_loss_instance'] = val_losses.avg
                state['mode'] = 'instance'
                
                save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, ops)
                save_metrics(config, state, train_pred, val_pred)
                print("Saved checkpoint due to improved val_loss_instance")
        
        
        state['warmup'] = False
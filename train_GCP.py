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
from archs.model_instances_tiny import *
from data.bag_loader import *
from data.instance_loader import *
from loss.FocalLoss import *
from util.eval_util import *
from config import *

torch.backends.cudnn.benchmark = True

# Instead of fixing device to cuda:0, just use 'cuda' if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optional: This is often useful for debugging but can degrade performance on multi-GPU setups
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

if __name__ == '__main__':
    print(torch.cuda.device_count(), "GPUs detected.")
    print(f'device: {device}')
    # Config
    model_version = '1'
    head_name = "TEST937192"
    data_config = LesionDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, _, _ = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes=num_labels)
    model = model.to(device)  # Move model to device

    # Wrap model in DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = optim.SGD(model.parameters(),
                          lr=config['learning_rate'],
                          momentum=0.9,
                          nesterov=True,
                          weight_decay=0.001)  # original .001

    BCE_loss = nn.BCELoss()

    # MODEL INIT
    # Pass the wrapped model and optimizer to any load/checkpoint function
    model, optimizer, state = setup_model(model, config, optimizer)
    
    # Use the instance predictions from bag training to update the Instance Dataloader
    instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=True)
    instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True)

    train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
    val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], strategy=1)

    instance_dataloader_train = TUD.DataLoader(
        instance_dataset_train,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=collate_instance,
        #persistent_workers=True,
        #pin_memory=True
    )

    instance_dataloader_val = TUD.DataLoader(
        instance_dataset_val,
        batch_sampler=val_sampler,
        collate_fn=collate_instance
    )

    print('Training Feature Extractor')
    print(f'Warmup Mode: {state["warmup"]}')

    for iteration in range(200): 
        losses = AverageMeter()
        instance_total_correct = 0
        total_samples = 0
        model.train()
        train_pred = PredictionTracker()
        
        for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
            images = images.to(device, non_blocking=True)
            instance_labels = instance_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Forward pass
            # If using DataParallel, model returns a tuple the same way it does on single-GPU
            _, _, instance_predictions, features = model(images)
            
            # Calculate loss
            total_loss = BCE_loss(instance_predictions, instance_labels.float())
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Update metrics
            losses.update(total_loss.item(), images.size(0))
            
            predicted = (instance_predictions > 0.5).float()
            correct = (predicted == instance_labels).float()
            instance_total_correct += correct.sum().item()
            total_samples += instance_labels.numel()
            
            # Store raw predictions and targets
            train_pred.update(instance_predictions, instance_labels, unique_id)

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
                images = images.to(device, non_blocking=True)
                instance_labels = instance_labels.to(device, non_blocking=True)

                # Forward pass
                _, _, instance_predictions, features = model(images)
                
                # Get loss
                loss_value = BCE_loss(instance_predictions, instance_labels.float())
                total_loss = loss_value
                val_losses.update(total_loss.item(), images.size(0))

                # Calculate correct predictions
                predicted = (instance_predictions > 0.5).float()  # Using 0.5 as threshold
                correct = (predicted == instance_labels).float()
                instance_total_correct += correct.sum().item()
                total_samples += instance_labels.numel()
                
                # Store raw predictions and targets
                val_pred.update(instance_predictions, instance_labels, unique_id)

        # Calculate accuracies
        instance_val_acc = instance_total_correct / total_samples
        
        print(f'[{iteration+1}/200] Train Loss: {losses.avg:.5f}, Train FC Acc: {instance_train_acc:.5f}')
        print(f'[{iteration+1}/200] Val Loss:   {val_losses.avg:.5f}, Val FC Acc: {instance_val_acc:.5f}')
        
        # Save the model
        if val_losses.avg < state['val_loss_instance']:
            state['val_loss_instance'] = val_losses.avg
            state['mode'] = 'instance'
            
            save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, optimizer)
            save_metrics(config, state, train_pred, val_pred)
            print("Saved checkpoint due to improved val_loss_instance")

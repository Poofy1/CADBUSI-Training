import os
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.sudo_labels import *
from archs.model_PALM2_solo import *
from data.bag_loader import *
from data.instance_loader import *
from config import *
from util.eval_util import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "Instance_testing_mixup_effi_3"
    data_config = LesionDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])


    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    
    # LOSS INIT
    BCE_loss = nn.BCELoss()

    
    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, config, optimizer)
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        # Used the instance predictions from bag training to update the Instance Dataloader
        instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=True)
        instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True)
        train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
        val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], strategy=1)
        instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=4, collate_fn = collate_instance, pin_memory=True)
        instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
        
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
            
            # Initialize lists to store training predictions and targets
            train_pred = []
            train_targets = []
            
            # Iterate over the training data
            for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                images = images.cuda(non_blocking=True)
                instance_labels = instance_labels.cuda(non_blocking=True)

                # Apply mixup
                lam = np.random.beta(alpha, alpha)  # alpha is typically 0.2 or 1.0
                batch_size = images.size()[0]
                index = torch.randperm(batch_size).cuda()
                
                # Mix the images
                mixed_images = lam * images + (1 - lam) * images[index, :]
                
                # Mix the labels
                labels_a, labels_b = instance_labels, instance_labels[index]

                # Forward pass
                optimizer.zero_grad()
                _, _, instance_predictions, features = model(mixed_images, projector=True)
                features = features.to(device)

                # Calculate mixed BCE loss
                loss_a = BCE_loss(instance_predictions, labels_a.float())
                loss_b = BCE_loss(instance_predictions, labels_b.float())
                bce_loss_value = lam * loss_a + (1 - lam) * loss_b

                # Backward pass and optimization
                total_loss = bce_loss_value
                total_loss.backward()
                optimizer.step()

                # Update metrics
                losses.update(total_loss.item(), images.size(0))
                
                # Calculate accuracy considering both original and mixed labels
                predicted = (instance_predictions > 0.5).float()
                correct_a = (predicted == labels_a).float()
                correct_b = (predicted == labels_b).float()
                
                # Weight the correctness by mixup lambda
                correct = lam * correct_a + (1 - lam) * correct_b
                instance_total_correct += correct.sum().item()
                total_samples += instance_labels.numel()
                
                # Store raw predictions and targets
                train_pred.append(instance_predictions.cpu().detach())
                train_targets.append(instance_labels.cpu().detach())

            # Calculate training accuracy
            instance_train_acc = instance_total_correct / total_samples
            
            # Concatenate all training predictions and targets
            train_pred = torch.cat(train_pred, dim=0)
            train_targets = torch.cat(train_targets, dim=0)
            
            # Validation loop
            model.eval()
            instance_total_correct = 0
            total_samples = 0
            val_losses = AverageMeter()
            
            # Initialize lists to store validation predictions and targets
            val_pred = []
            val_targets = []

            with torch.no_grad():
                for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)

                    # Forward pass
                    _, _, instance_predictions, features = model(images, projector=True)
                    features.to(device)
                    
                    # Get loss
                    bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())
                    total_loss = bce_loss_value
                    val_losses.update(total_loss.item(), images[0].size(0))
                    
                    # Calculate correct predictions
                    predicted = (instance_predictions > 0.5).float()
                    correct = (predicted == instance_labels).float()
                    instance_total_correct += correct.sum().item()
                    total_samples += instance_labels.numel()
                    
                    # Store raw predictions and targets
                    val_pred.append(instance_predictions.cpu().detach())
                    val_targets.append(instance_labels.cpu().detach())

            # Calculate validation accuracy
            instance_val_acc = instance_total_correct / total_samples
            
            # Concatenate all validation predictions and targets
            val_pred = torch.cat(val_pred, dim=0)
            val_targets = torch.cat(val_targets, dim=0)
            
            print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train FC Acc: {instance_train_acc:.5f}')
            print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}, Val FC Acc: {instance_val_acc:.5f}')
            
            save_metrics(config, state, train_targets, train_pred, val_targets, val_pred, mode = 'instance')
                    
            # Save the model
            if val_losses.avg < state['val_loss_instance']:
                state['val_loss_instance'] = val_losses.avg
                
                save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, optimizer)
                print("Saved checkpoint due to improved val_loss_instance")
        
        
        state['warmup'] = False
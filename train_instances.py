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
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "Instance_testing"
    data_config = FishDataConfig  # or LesionDataConfig
    
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
    model, optimizer, state = setup_model(model, optimizer, config)
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        if True: # Are we resuming from a head model?
        
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
            
            # Unfreeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = True

            
            
            for iteration in range(target_count): 
                losses = AverageMeter()
                instance_total_correct = 0
                total_samples = 0
                model.train()
                
                # Iterate over the training data
                for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)

                    # forward
                    optimizer.zero_grad()
                    _, _, instance_predictions, features = model(images, projector=True)
                    features = features.to(device)
                    
                    # Calculate BCE loss
                    bce_loss_value = BCE_loss(instance_predictions, instance_labels.float())

                    # Backward pass and optimization step
                    total_loss = bce_loss_value
                    total_loss.backward()
                    optimizer.step()

                    # Update the loss meter
                    losses.update(total_loss.item(), images.size(0))
                    
                    # Calculate correct predictions
                    predicted = (instance_predictions > 0.5).float()  # Assuming 0.5 as threshold
                    correct = (predicted == instance_labels).float()
                    instance_total_correct += correct.sum().item()
                    total_samples += instance_labels.numel()

                # Calculate accuracies
                instance_train_acc = instance_total_correct / total_samples
                                
                
                
                # Validation loop
                model.eval()
                instance_total_correct = 0
                total_samples = 0
                val_losses = AverageMeter()

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
                        predicted = (instance_predictions > 0.5).float()  # Assuming 0.5 as threshold
                        correct = (predicted == instance_labels).float()
                        instance_total_correct += correct.sum().item()
                        total_samples += instance_labels.numel()

                # Calculate accuracies
                instance_val_acc = instance_total_correct / total_samples
                
                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train FC Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}, Val FC Acc: {instance_val_acc:.5f}')
                
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    if state['warmup']:
                        target_folder = state['head_folder']
                        target_name = state['pretrained_name']
                    else:
                        target_folder = state['model_folder']
                        target_name = state['model_name']
                    all_targs = []
                    all_preds = []
                    
                    if state['warmup']:
                        save_state(state['epoch'], config['label_columns'], instance_train_acc, val_losses.avg, instance_val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                        print("Saved checkpoint due to improved val_loss_instance")





        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            


import os, pickle
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.pseudo_labels import *
from data.instance_loader import *
from loss.SupCon import SupConLoss
from loss.SimCLR import SimCLRLoss
from util.eval_util import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import gc


if __name__ == '__main__':
    
    config = build_config()
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    model = build_model(config)    
    
    # LOSS INIT
    loss = SimCLRLoss()
    
    ops = {}
    ops['inst_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
    ops['bag_optimizer'] = optim.Adam(model.parameters(),
                        lr=config['learning_rate'],
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0.001)

    # MODEL INIT
    model, ops, state = setup_model(model, config, ops)
    scaler = GradScaler('cuda')
    
    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataloader_train, instance_dataloader_val = get_instance_loaders(bags_train, bags_val, 
                                                                                      state, config, 
                                                                                      use_bag_labels=True, dual_output = True)
            
            if state['warmup']:
                target_count = config['warmup_epochs']
            else:
                target_count = config['feature_extractor_train_count']
                
                
            
            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            
            
            for iteration in range(target_count): 
                losses = AverageMeter()
                instance_total_correct = 0
                total_samples = 0
                train_pred = PredictionTracker()
                model.train()
                
                # Iterate over the training data
                for idx, (images, instance_labels, pseudo_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    bsz = instance_labels.shape[0]
                    im_q, im_k = images
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
                    images = [im_q, im_k]
                    images = torch.cat(images, dim=0).cuda()
                    
                    # forward
                    ops['inst_optimizer'].zero_grad()

                    with autocast('cuda'):
                        _, instance_predictions, features = model(images, projector=True)

                    zk, zq = torch.split(features, [bsz, bsz], dim=0)
                    features_for_supcon = torch.stack([zk, zq], dim=1)
                    
                    # Calculate loss
                    total_loss = loss(zk, zq)

                    # Backward pass and optimization step
                    scaler.scale(total_loss).backward()
                    scaler.step(ops['inst_optimizer'])
                    scaler.update()
        
                    # Update the loss meter
                    losses.update(total_loss.item(), bsz)
                        
                    # Store raw predictions and targets
                    #train_pred.update(instance_predictions, instance_labels, unique_id)
                    total_samples += instance_labels.size(0)
                    
                    # Clean up
                    torch.cuda.empty_cache()

                # Calculate accuracies
                instance_train_acc = instance_total_correct / total_samples
                                
                
                
                # Validation loop
                model.eval()
                instance_total_correct = 0
                total_samples = 0
                val_losses = AverageMeter()
                val_pred = PredictionTracker()

                with torch.no_grad():
                    for idx, (images, instance_labels, pseudo_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        bsz = instance_labels.shape[0]
                        im_q, im_k = images
                        im_q = im_q.cuda(non_blocking=True)
                        im_k = im_k.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)
                        images = [im_q, im_k]
                        images = torch.cat(images, dim=0).cuda()

                        # Forward pass
                        with autocast('cuda'):
                            _, instance_predictions, features = model(images, projector=True)
                        
                        zk, zq = torch.split(features, [bsz, bsz], dim=0)
                        features_for_supcon = torch.stack([zk, zq], dim=1)
                        
                        # Calculate loss
                        total_loss = loss(zk, zq)

                        # Store raw predictions and targets
                        #val_pred.update(instance_predictions, instance_labels, unique_id)
                        val_losses.update(total_loss.item(), bsz)
                        total_samples += instance_labels.size(0)
                        
                        # Clean up
                        torch.cuda.empty_cache()

                # Calculate accuracies
                instance_val_acc = instance_total_correct / total_samples
                
                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}')
                
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    state['mode'] = 'instance'
                    
                    if state['warmup']:
                        target_folder = state['head_folder']
                    else:
                        target_folder = state['model_folder']
                    
                    #save_metrics(config, state, train_pred, val_pred)
                    
                    if state['warmup']:
                        save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, ops)
                        print("Saved checkpoint due to improved val_loss_instance")





        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            
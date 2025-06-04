import os, pickle
from fastai.vision.all import *
import torch.optim as optim
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
from torch.optim import Adam
from data.format_data import *
from data.pseudo_labels import *
from data.bag_loader import *
from data.instance_loader import *
from loss.genSCL import GenSupConLossv2
from util.eval_util import *
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
from torch.amp import autocast, GradScaler
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import gc


if __name__ == '__main__':
    
    mix_alpha=0.2
    mix='mixup'
    
    config = build_config()
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    model = build_model(config)
        
    BCE_loss = nn.BCEWithLogitsLoss()
    genscl = GenSupConLossv2(temperature=0.07, base_temperature=0.07)


    ops = {}
    ops['inst_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
    ops['bag_optimizer'] = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
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
                                                                                      warmup=True, dual_output=True)
            
            if state['warmup']:
                target_count = config['warmup_epochs']
            else:
                target_count = config['feature_extractor_train_count']
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            

            # Generalized Supervised Contrastive Learning phase
            
            model.train()
            for iteration in range(target_count): 
                losses = AverageMeter()
                

                # Iterate over the training data
                for idx, (images, instance_labels, unique_ids) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    #warmup_learning_rate(args, epoch, idx, len(instance_dataloader_train), optimizer)
                    

                    # Data preparation 
                    bsz = instance_labels.shape[0]
                    im_q, im_k = images
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
                    
                    # image-based regularizations (lam 1 = no mixup)
                    im_q, y0a, y0b, lam0 = mix_fn(im_q, instance_labels, mix_alpha, mix)
                    im_k, y1a, y1b, lam1 = mix_fn(im_k, instance_labels, mix_alpha, mix)
                    
                    l_q = mix_target(y0a, y0b, lam0, config['num_classes'])
                    l_k = mix_target(y1a, y1b, lam1, config['num_classes'])
                    
                    # forward
                    ops['inst_optimizer'].zero_grad()
                    
                    images = [im_q, im_k]
                    all_images = torch.cat(images, dim=0).cuda()

                    # Forward pass
                    with autocast('cuda'):
                        _, _, _, features = model(all_images, projector=True)
                        features = features.cuda()
                        
                    zk, zq = torch.split(features, [bsz, bsz], dim=0)
                    
                    # get loss (no teacher)
                    mapped_anchors = ~(instance_labels == -1).bool()
                    loss = genscl([zk, zq], [l_q, l_k], (mapped_anchors, mapped_anchors))
                    losses.update(loss.item(), bsz)

                    scaler.scale(loss).backward()
                    scaler.step(ops['inst_optimizer'])
                    scaler.update()
                    
                print(f'[{iteration+1}/{target_count}] Gen_SCL Loss: {losses.avg:.5f}')



        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            
            print("Saved Warmup Model")
            torch.save(model.state_dict(), os.path.join(state['head_folder'], f"{state['pretrained_name']}.pth"))
            torch.save(ops['inst_optimizer'].state_dict(), f"{state['head_folder']}/{state['pretrained_name']}_optimizer.pth")
            
            
        

        
        print('\nTraining Bag Aggregator')
        for iteration in range(config['MIL_train_count']):
            model.train()
            train_bag_logits = {}
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0
            train_pred = PredictionTracker()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for batch_idx, (images, yb, instance_labels, unique_id) in enumerate(tqdm(bag_dataloader_train, total=len(bag_dataloader_train))):
                num_bags = len(images)
                ops['bag_optimizer'].zero_grad()
                #images, yb = images.cuda(), yb.cuda()

                if not isinstance(images, list):
                    # If images is a padded tensor
                    images, yb = images.cuda(), yb.cuda()
                else:
                    # If images is a list of tensors
                    images = [img.cuda() for img in images]
                    yb = yb.cuda()
        
                # Forward pass
                with autocast('cuda'):
                    bag_pred, bag_instance_pred, _, _ = model(images, pred_on=True)
                    bag_pred = bag_pred.cuda()
                    
                # Split the embeddings back into per-bag embeddings
                split_sizes = []
                for bag in images:
                    # Remove padded images (assuming padding is represented as zero tensors)
                    valid_images = bag[~(bag == 0).all(dim=1).all(dim=1).all(dim=1)] # Shape: [valid_images, 224, 224, 3]
                    split_sizes.append(valid_images.size(0))

                #instance_pred = torch.cat(instance_pred, dim=0)
                y_hat_per_bag = torch.split(torch.sigmoid(bag_instance_pred), split_sizes, dim=0)
                for i, y_h in enumerate(y_hat_per_bag):
                    train_bag_logits[unique_id[i].item()] = y_h.detach().cpu().numpy()
                
                bag_loss = BCE_loss(bag_pred, yb)
                scaler.scale(bag_loss).backward()
                scaler.step(ops['bag_optimizer'])
                scaler.update()
                
                bag_pred = torch.sigmoid(bag_pred)
                total_loss += bag_loss.item() * yb.size(0)
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
                # Store raw predictions and targets
                train_pred.update(bag_pred, yb, unique_id)
                
                
                ### It seems that GCP requires this cleanup?
                
                # Make sure we're explicitly cleaning up
                if isinstance(images, list):
                    for img in images:
                        img.detach()
                        del img
                else:
                    images.detach()
                    del images

                del bag_instance_pred
                del y_hat_per_bag
                del bag_pred

                # Clean up
                torch.cuda.empty_cache()
                gc.collect()
                    
            
            
            train_loss = total_loss / total
            train_acc = correct / total
                    
                    
            # Evaluation phase
            model.eval()
            total = 0
            correct = 0
            total_val_loss = 0.0
            total_val_acc = 0.0
            val_pred = PredictionTracker()

            with torch.no_grad():
                for (images, yb, instance_labels, unique_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                    if not isinstance(images, list):
                        # If images is a padded tensor
                        images, yb = images.cuda(), yb.cuda()
                    else:
                        # If images is a list of tensors
                        images = [img.cuda() for img in images]
                        yb = yb.cuda()
                        
                        
                    # Forward pass
                    with autocast('cuda'):
                        bag_pred, _, _, features = model(images, pred_on=True)
                        bag_pred = bag_pred.cuda()

                    # Calculate bag-level loss
                    loss = BCE_loss(bag_pred, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    bag_pred = torch.sigmoid(bag_pred)
                    predicted = (bag_pred > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()

                    # Store raw predictions and targets
                    val_pred.update(bag_pred, yb, unique_id)
                    
                    # Clean up
                    torch.cuda.empty_cache()
                        
            val_loss = total_val_loss / total
            val_acc = correct / total
                
        
            state['train_losses'].append(train_loss)
            state['valid_losses'].append(val_loss)    
            
            print(f"[{iteration+1}/{config['MIL_train_count']}] | Acc | Loss")
            print(f"Train | {train_acc:.4f} | {train_loss:.4f}")
            print(f"Val | {val_acc:.4f} | {val_loss:.4f}")

            # Save the model
            if val_loss < state['val_loss_bag']:
                state['val_loss_bag'] = val_loss
                state['mode'] = 'bag'
                if state['warmup']:
                    target_folder = state['head_folder']
                else:
                    target_folder = state['model_folder']

                
                save_state(state, config, train_acc, val_loss, val_acc, model, ops,)
                save_metrics(config, state, train_pred, val_pred)
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1
                
                # Create selection mask
                predictions_ratio = prediction_anchor_scheduler(state['epoch'], config)
                state['selection_mask'] = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new pseudo-labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)

                    
                    
                    
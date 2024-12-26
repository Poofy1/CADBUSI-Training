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
from data.sudo_labels import *
#from archs.model_GenSCL import *
from archs.model_solo_MIL import *
from data.bag_loader import *
from data.instance_loader import *
from loss.genSCL import GenSupConLossv2
from util.eval_util import *
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    


if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "TEST35"
    data_config = FishDataConfig  # or LesionDataConfig
    
    mix_alpha=0.2
    mix='mixup'
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")    
        
    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001
    BCE_loss = nn.BCELoss()
    genscl = GenSupConLossv2(temperature=0.07, base_temperature=0.07)


    model, optimizer, state = setup_model(model, config, optimizer)


    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=True, dual_output=True)
            
            if state['warmup']:
                sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'])
                instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=sampler, collate_fn = collate_instance)
                target_count = config['warmup_epochs']
            else:
                instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_size=config['instance_batch_size'], collate_fn = collate_instance, drop_last=True, shuffle = True)
                target_count = config['feature_extractor_train_count']
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            
            # Unfreeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = True
        
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
                    images = [im_q, im_k]
                    l_q = mix_target(y0a, y0b, lam0, num_classes)
                    l_k = mix_target(y1a, y1b, lam1, num_classes)
                    
                    # forward
                    optimizer.zero_grad()

                    all_images = torch.cat(images, dim=0).cuda()  # TEMP FOR TESTING
                    _, _, _, features = model(all_images, projector=True)
                    zk, zq = torch.split(features, [bsz, bsz], dim=0)
                    
                    # get loss (no teacher)
                    mapped_anchors = ~(instance_labels == -1).bool()
                    loss = genscl([zk, zq], [l_q, l_k], (mapped_anchors, mapped_anchors))
                    losses.update(loss.item(), bsz)

                    loss.backward()
                    optimizer.step()
                    
                print(f'[{iteration+1}/{target_count}] Gen_SCL Loss: {losses.avg:.5f}')



        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            
            print("Saved Warmup Model")
            torch.save(model.state_dict(), os.path.join(state['head_folder'], f"{state['pretrained_name']}.pth"))
            torch.save(optimizer.state_dict(), f"{state['head_folder']}/{state['pretrained_name']}_optimizer.pth")
            
            
        
        
        if config['reset_aggregator']:
            model.aggregator.reset_parameters() # Reset the model.aggregator weights before training
        
        # Freeze the encoder
        """for param in model.encoder.parameters():
            param.requires_grad = False"""
        
        # Training phase
        print('\nTraining Bag Aggregator')
        for iteration in range(config['MIL_train_count']):
            
            model.train()
            total_loss = 0.0
            train_bag_logits = {}
            total_acc = 0
            total = 0
            correct = 0
            train_pred = PredictionTracker()

            for (data, yb, instance_yb, unique_id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
                xb, yb = data, yb.cuda()
            
                optimizer.zero_grad()
                
                bag_pred, instance_predictions, _, _ = model(xb, pred_on = True)
                #print(outputs)
                #print(yb)
                bag_pred = torch.clamp(bag_pred, 0, 1) # temp fix
                
                # Calculate bag-level loss
                loss = BCE_loss(bag_pred, yb)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * yb.size(0)
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
                for instance_id, bag_id in enumerate(unique_id):
                    train_bag_logits[bag_id] = instance_predictions[instance_id].detach().cpu().numpy()

                # Store raw predictions and targets
                train_pred.update(bag_pred, yb, unique_id)
                
            train_loss = total_loss / total
            train_acc = correct / total

            # Evaluation phase
            model.eval()
            total_val_loss = 0.0
            total_val_acc = 0.0
            total = 0
            correct = 0
            val_pred = PredictionTracker()

            with torch.no_grad():
                for (data, yb, instance_yb, unique_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                    xb, yb = data, yb.cuda()

                    bag_pred, instance_predictions, _, _ = model(xb, pred_on = True)
                    #print(instance_pred)
                    bag_pred = torch.clamp(bag_pred, 0, 1) # temp fix

                    # Calculate bag-level loss
                    loss = BCE_loss(bag_pred, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    predicted = (bag_pred > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()

                    # Store raw predictions and targets
                    val_pred.update(bag_pred, yb, unique_id)
            

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

                save_state(state, config, train_acc, val_loss, val_acc, model, optimizer,)
                save_metrics(config, state, train_pred, val_pred)
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1
                
                # Create selection mask
                predictions_ratio = prediction_anchor_scheduler(state['epoch'], config['total_epochs'], 0, config['initial_ratio'], config['final_ratio'])
                #predictions_ratio = .9
                state['selection_mask'] = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                if state['warmup']:
                    target_folder = state['head_folder']
                else:
                    target_folder = state['model_folder']
                    
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)

                    
                    
                    
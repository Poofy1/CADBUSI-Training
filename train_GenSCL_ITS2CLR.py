import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
from torch.optim import Adam
from data.format_data import *
from data.sudo_labels import *
from archs.model_GenSCL import *
from data.bag_loader import *
from data.instance_loader import *
from loss.genSCL import GenSupConLossv2
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    


if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "GenSCL_OFFICAL"
    data_config = FishDataConfig  # or LesionDataConfig
    
    mix_alpha=0.2
    mix='mixup'
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create bag datasets
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
    bag_dataset_val = BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False)
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=config['bag_batch_size'], collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_size=config['bag_batch_size'], collate_fn = collate_bag, drop_last=True)


    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")    
        
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    BCE_loss = nn.BCELoss()
    genscl = GenSupConLossv2(temperature=0.07, base_temperature=0.07)


    model, optimizer, state = setup_model(model, optimizer, config)


    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=state['warmup'], dual_output=True)
            
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
                    _, _, features = model(images, projector=True)
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
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        # Training phase
        print('\nTraining Bag Aggregator')
        for iteration in range(config['MIL_train_count']):
            
            model.train()
            total_loss = 0.0
            train_bag_logits = {}
            total_acc = 0
            total = 0
            correct = 0

            for (data, yb, instance_yb, id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
                xb, yb = data, yb.cuda()
            
                optimizer.zero_grad()
                
                outputs, instance_pred, _ = model(xb, pred_on = True)
                #print(outputs)
                #print(yb)
                outputs = torch.clamp(outputs, 0, 1) # temp fix
                
                # Calculate bag-level loss
                loss = BCE_loss(outputs, yb)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * yb.size(0)
                predicted = (outputs > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
                for instance_id, bag_id in enumerate(id):
                    train_bag_logits[bag_id] = instance_pred[instance_id].detach().cpu().numpy()

            train_loss = total_loss / total
            train_acc = correct / total

            # Evaluation phase
            model.eval()
            total_val_loss = 0.0
            total_val_acc = 0.0
            total = 0
            correct = 0
            all_targs = []
            all_preds = []

            with torch.no_grad():
                for (data, yb, instance_yb, id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                    xb, yb = data, yb.cuda()

                    outputs, instance_pred, _ = model(xb, pred_on = True)
                    #print(instance_pred)
                    outputs = torch.clamp(outputs, 0, 1) # temp fix

                    # Calculate bag-level loss
                    loss = BCE_loss(outputs, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    predicted = (outputs > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()

                    # Confusion Matrix data
                    all_targs.extend(yb.cpu().numpy())
                    if len(predicted.size()) == 0:
                        predicted = predicted.view(1)
                    all_preds.extend(predicted.cpu().detach().numpy())
            

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
                if state['warmup']:
                    target_folder = state['head_folder']
                    target_name = state['pretrained_name']
                else:
                    target_folder = state['model_folder']
                    target_name = state['model_name']
                
                save_state(state['epoch'], config['label_columns'], train_acc, val_loss, val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                print("Saved checkpoint due to improved val_loss_bag")
                
                # Create selection mask
                predictions_ratio = prediction_anchor_scheduler(state['epoch'], config['total_epochs'], 0, config['initial_ratio'], config['final_ratio'])
                #predictions_ratio = .9
                state['selection_mask'] = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                state['epoch'] += 1
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)

                    
                    
                    
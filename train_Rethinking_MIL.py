import os
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.sudo_labels import *
from archs.model_INS import *
from loss.IWSCL import *
from data.bag_loader import *
from data.instance_loader import *
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        
        
if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "TEST_R_MIL"
    data_config = FishDataConfig  # or LesionDataConfig
    
    
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
    
    # LOSS INIT
    BCE_loss = nn.BCELoss()
    CE_crit = nn.CrossEntropyLoss()
    IWSCL_crit = IWSCL(128).to(device)
    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, optimizer, config)

    
    # Training loop
    while state['epoch'] < config['total_epochs']:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=state['warmup'], dual_output=True)
            instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True, dual_output=True)
            train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
            val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], strategy=1)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=2, collate_fn = collate_instance, pin_memory=True)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
            if state['warmup']:
                target_count = config['warmup_epochs']
            else:
                target_count = config['feature_extractor_train_count']
            
            
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')

            
            
            for iteration in range(target_count): 
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                model.train()
                
                train_supcon_loss_total = AverageMeter()
                train_ce_loss_total = AverageMeter()
                
                # Iterate over the training data
                for idx, ((im_q, im_k), instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)

                    # forward
                    optimizer.zero_grad()
                    _, _, instance_predictions, feat_q = model(im_q, im_k, true_label = instance_labels, projector=True)
                    feat_q.to(device)
                    
                    
                    # Calculate loss
                    queue, queue_labels = model.get_queue()
                    supcon_loss, pseudo_labels = IWSCL_crit(feat_q, instance_predictions, instance_labels, queue, queue_labels)
                    ce_loss = CE_crit(instance_predictions, instance_labels.float())
                    total_loss = ce_loss + supcon_loss
                    
                    # Backward pass and optimization step
                    total_loss.backward()
                    optimizer.step()

                    # Update the loss meter
                    train_supcon_loss_total.update(supcon_loss.item(), instance_labels.size(0))
                    train_ce_loss_total.update(ce_loss.item(), instance_labels.size(0))
                    
                    # Get predictions
                    with torch.no_grad():
                        # Create mask for valid labels (0 and 1)
                        valid_mask = (instance_labels != -1)
                        
                        # Apply mask to get predictions and labels
                        valid_predictions = instance_predictions[valid_mask]
                        valid_labels = instance_labels[valid_mask]
                        
                        # Calculate accuracy for valid samples only
                        instance_predicted_classes = (valid_predictions > 0.5)
                        instance_correct = (instance_predicted_classes == valid_labels).sum().item()
                        instance_total_correct += instance_correct
                        total_samples += valid_mask.sum().item()

                # Calculate accuracies
                palm_train_acc = palm_total_correct / total_samples
                instance_train_acc = instance_total_correct / total_samples
                                
                
                
                # Validation loop
                model.eval()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                val_supcon_loss_total = AverageMeter()
                val_ce_loss_total = AverageMeter()
                val_losses = AverageMeter()

                with torch.no_grad():
                    for idx, ((im_q, im_k), instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        im_q = im_q.cuda(non_blocking=True)
                        im_k = im_k.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)

                        # Forward pass
                        _, _, instance_predictions, feat_q = model(im_q, projector=True)
                        feat_q.to(device)
                        
                        # Calculate loss
                        queue, queue_labels = model.get_queue()
                        supcon_loss, pseudo_labels = IWSCL_crit(feat_q, instance_predictions, instance_labels, queue, queue_labels, val_on = True)
                        ce_loss = CE_crit(instance_predictions, pseudo_labels.float())
                        total_loss = ce_loss + supcon_loss
                        
                        val_supcon_loss_total.update(supcon_loss.item(), instance_labels.size(0))
                        val_ce_loss_total.update(ce_loss.item(), instance_labels.size(0))
                        val_losses.update(total_loss.item(), instance_labels.size(0))

                        # Get predictions
                        instance_predicted_classes = (instance_predictions) > 0.5
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        total_samples += instance_labels.size(0)

                # Calculate accuracies
                palm_val_acc = palm_total_correct / total_samples
                instance_val_acc = instance_total_correct / total_samples
                
                print(f'[{iteration+1}/{target_count}] Train | Supcon Loss: {train_supcon_loss_total.avg:.5f}, CE Loss: {train_ce_loss_total.avg:.5f}, Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val   | Supcon Loss: {val_supcon_loss_total.avg:.5f}, CE Loss: {val_ce_loss_total.avg:.5f}, Acc: {instance_val_acc:.5f}')
                
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
            

        
            
        print('\nTraining Bag Aggregator')
        for iteration in range(config['MIL_train_count']):
            model.train()
            train_bag_logits = {}
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0

            for (images, yb, instance_labels, id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
                optimizer.zero_grad()

                # Forward pass
                bag_pred, _, instance_pred, _ = model(images, bag_on=True)

                bag_loss = BCE_loss(bag_pred, yb)
                bag_loss.backward()
                optimizer.step()
                
                total_loss += bag_loss.item() * yb.size(0)
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                    
            
            
            train_loss = total_loss / total
            train_acc = correct / total
                    
                    
            # Evaluation phase
            model.eval()
            total = 0
            correct = 0
            total_val_loss = 0.0
            total_val_acc = 0.0
            all_targs = []
            all_preds = []

            with torch.no_grad():
                for (images, yb, instance_labels, id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 

                    # Forward pass
                    bag_pred, _, _, _ = model(images, bag_on=True, val_on = True)

                    # Calculate bag-level loss
                    loss = BCE_loss(bag_pred, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    predicted = (bag_pred > 0.5).float()
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

                
                state['epoch'] += 1
                
                # Create selection mask
                """predictions_ratio = prediction_anchor_scheduler(state['epoch'], total_epochs, 0, initial_ratio, final_ratio)
                state['selection_mask'] = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)"""


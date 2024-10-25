import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.sudo_labels import *
from archs.model_PALM import *
from data.bag_loader import *
from data.instance_loader import *
from loss.palm import PALM
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    



if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "TEST_PALM_ITS2CLR"
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
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 90, lambda_pcon=0).cuda()
    BCE_loss = nn.BCELoss()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, optimizer, config)
    palm.load_state(state['palm_path'])
    
    # Training loop
    while state['epoch'] < config['total_epochs']:

        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform)
            instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform)
            train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
            val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], strategy=2)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, collate_fn = collate_instance)
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
                total_correct = 0
                total_samples = 0
                model.train()
                
                # Iterate over the training data
                for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
  
                    # forward
                    optimizer.zero_grad()
                    _, _, features = model(images, projector=True)
                    features.to(device)

                    # Get loss from PALM
                    loss, loss_dict = palm(features, instance_labels)

                    # Backward pass and optimization step
                    loss.backward()
                    optimizer.step()
        
                    # Update the loss meter
                    losses.update(loss.item(), images[0].size(0))
                    
                    # Get predictions from PALM
                    with torch.no_grad():
                        predicted_classes, dist = palm.predict(features)

                    # Calculate accuracy
                    correct = (predicted_classes == instance_labels).sum().item()
                    total_correct += correct
                    total_samples += instance_labels.size(0)

                instance_train_acc = total_correct / total_samples
                
                
                
                # Validation loop
                model.eval()
                val_losses = AverageMeter()
                val_total_correct = 0
                val_total_samples = 0

                with torch.no_grad():
                    for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        images = images.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)

                        # Forward pass
                        _, _, features = model(images, projector=True)
                        features.to(device)
                        
                        # Get loss
                        total_loss, loss_dict = palm(features, instance_labels, update_prototypes=False)
                        val_losses.update(total_loss.item(), images[0].size(0))

                        # Get predictions from PALM
                        predicted_classes, _ = palm.predict(features)

                        # Calculate accuracy
                        correct = (predicted_classes == instance_labels).sum().item()
                        val_total_correct += correct
                        val_total_samples += instance_labels.size(0)

                instance_val_acc = val_total_correct / val_total_samples
                
                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}, Val Acc:   {instance_val_acc:.5f}')

                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    
                    if state['warmup']:
                        save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, optimizer)
                        palm.save_state(os.path.join(target_folder, "palm_state.pkl"))
                        print("Saved checkpoint due to improved val_loss_instance")



        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
        
        
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
                else:
                    target_folder = state['model_folder']

                
                save_state(state, config, train_acc, val_loss, val_acc, model, optimizer, all_targs, all_preds)
                palm.save_state(os.path.join(target_folder, "palm_state.pkl"))
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1
                
                # Create selection mask
                predictions_ratio = prediction_anchor_scheduler(state['epoch'], config['total_epochs'], 0, config['initial_ratio'], config['final_ratio'])
                #predictions_ratio = .9
                state['selection_mask'] = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)


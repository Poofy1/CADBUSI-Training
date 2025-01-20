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
from archs.model_solo_MIL_saliency import *
from data.bag_loader import *
from data.instance_loader import *
from loss.palm import PALM
from util.eval_util import *
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "TEST120"
    data_config = FishDataConfig  # FishDataConfig or DogDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    
    # LOSS INIT
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 0, lambda_pcon=1).cuda()
    BCE_loss = nn.BCELoss()
    
    # Combine the parameters from both the embedding model and the PALM model
    opt_params = list(model.parameters()) + list(palm.parameters())
    
    optimizer = optim.SGD(opt_params,
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    

    # MODEL INIT
    model, optimizer, state = setup_model(model, config, optimizer)
    palm.load_state(state['palm_path'])
    
    # Training loop
    while state['epoch'] < config['total_epochs']:

        if not state['pickup_warmup']: # Are we resuming from a head model?
            
            
            torch.cuda.empty_cache()
            #state['selection_mask']
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=True)
            instance_dataset_val = Instance_Dataset(bags_val, [], transform=val_transform)
            train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, collate_fn = collate_instance)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_size=config['instance_batch_size'], collate_fn = collate_instance)
            
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
                train_pred = PredictionTracker()
                
                # Initialize dictionaries to accumulate loss components
                train_loss_components = {}
                
                # Iterate over the training data
                for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
                    
                    # forward
                    optimizer.zero_grad()
                    _, _, _, features = model(images, projector=True)
                    features.to(device)

                    # Get loss from PALM
                    loss, loss_dict = palm(features, instance_labels)
                    
                    # Accumulate loss components
                    for key, value in loss_dict.items():
                        if key not in train_loss_components:
                            train_loss_components[key] = 0
                        train_loss_components[key] += value.item() if torch.is_tensor(value) else value
                    
                    # Backward pass and optimization step
                    loss.backward()
                    optimizer.step()

                    # Update the loss meter
                    losses.update(loss.item(), images[0].size(0))
                    
                    # Get predictions from PALM
                    with torch.no_grad():
                        instance_predictions, dist = palm.predict(features)

                    # Calculate accuracy
                    correct = (instance_predictions == instance_labels).sum().item()
                    total_correct += correct
                    total_samples += instance_labels.size(0)
                    
                    # Store raw predictions and targets
                    train_pred.update(instance_predictions, instance_labels, unique_id)
                    
                    # Clean up
                    torch.cuda.empty_cache()

                instance_train_acc = total_correct / total_samples
                
                # Validation loop
                model.eval()
                val_losses = AverageMeter()
                val_total_correct = 0
                val_total_samples = 0
                val_pred = PredictionTracker()
                
                # Initialize dictionaries to accumulate validation loss components
                val_loss_components = {}

                with torch.no_grad():
                    for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        images = images.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)

                        # Forward pass
                        _, _, _, features = model(images, projector=True)
                        features.to(device)
                        
                        # Get loss
                        total_loss, loss_dict = palm(features, instance_labels, update_prototypes=False)
                        
                        # Accumulate validation loss components
                        for key, value in loss_dict.items():
                            if key not in val_loss_components:
                                val_loss_components[key] = 0
                            val_loss_components[key] += value.item() if torch.is_tensor(value) else value
                            
                        val_losses.update(total_loss.item(), images[0].size(0))

                        # Get predictions from PALM
                        instance_predictions, _ = palm.predict(features)

                        # Calculate accuracy
                        correct = (instance_predictions == instance_labels).sum().item()
                        val_total_correct += correct
                        val_total_samples += instance_labels.size(0)
                        
                        # Store raw predictions and targets
                        val_pred.update(instance_predictions, instance_labels, unique_id)

                instance_val_acc = val_total_correct / val_total_samples
                
                # Calculate average loss components
                train_batch_count = len(instance_dataloader_train)
                val_batch_count = len(instance_dataloader_val)
                
                avg_train_components = {k: v/train_batch_count for k, v in train_loss_components.items()}
                avg_val_components = {k: v/val_batch_count for k, v in val_loss_components.items()}
                
                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train Acc: {instance_train_acc:.5f}')
                print('Training Loss Components:')
                for k, v in avg_train_components.items():
                    print(f'    {k}: {v:.5f}')
                    
                print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}, Val Acc:   {instance_val_acc:.5f}')
                print('Validation Loss Components:')
                for k, v in avg_val_components.items():
                    print(f'    {k}: {v:.5f}')
                        
        
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    state['mode'] = 'instance'
                    save_metrics(config, state, train_pred, val_pred)
                    
                    if state['warmup']:
                        target_folder = state['head_folder']
                    else:
                        target_folder = state['model_folder']
                    
                    if state['warmup']:
                        save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, optimizer)
                        palm.save_state(os.path.join(target_folder, "palm_state.pkl"))
                        print("Saved checkpoint due to improved val_loss_instance")


        """
        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
        
        # Clean up
        del instance_dataloader_train
        del instance_dataloader_val
        del instance_dataset_train
        del instance_dataset_val
        torch.cuda.empty_cache()
        
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
                
                bag_pred, instance_pred, _, features = model(xb, pred_on = True)
                
                # Calculate bag-level loss
                loss = BCE_loss(bag_pred, yb)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * yb.size(0)
                predicted = (bag_pred > 0.5).float()
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
                for instance_id, bag_id in enumerate(unique_id):
                    train_bag_logits[bag_id] = instance_pred[instance_id].detach().cpu().numpy()
                    
                # Store raw predictions and targets
                train_pred.update(bag_pred, yb, unique_id)
                
                torch.cuda.empty_cache()

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

                    bag_pred, instance_pred, _, features = model(xb, pred_on = True)
                    #print(instance_pred)

                    # Calculate bag-level loss
                    loss = BCE_loss(bag_pred, yb)
                    total_val_loss += loss.item() * yb.size(0)

                    predicted = (bag_pred > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted == yb).sum().item()

                    # Store raw predictions and targets
                    val_pred.update(bag_pred, yb, unique_id)
                    
                    torch.cuda.empty_cache()
            

            val_loss = total_val_loss / total
            val_acc = correct / total

            state['train_losses'].append(train_loss)
            state['valid_losses'].append(val_loss) 

            print(f"[{iteration+1}/{config['MIL_train_count']}] | Acc | Loss")
            print(f"Train | {train_acc:.4f} | {train_loss:.4f}")
            print(f"Val | {val_acc:.4f} | {val_loss:.4f}")
            
            
                        
            
            
            torch.cuda.empty_cache()

            # Save the model
            if val_loss < state['val_loss_bag']:
                state['val_loss_bag'] = val_loss
                state['mode'] = 'bag'
                
                if state['warmup']:
                    target_folder = state['head_folder']
                else:
                    target_folder = state['model_folder']

                
                save_state(state, config, train_acc, val_loss, val_acc, model, optimizer,)
                save_metrics(config, state, train_pred, val_pred)
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
                    pickle.dump(state['selection_mask'], file)"""
                    
                    
                    
            


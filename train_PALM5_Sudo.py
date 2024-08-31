import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from archs.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from util.format_data import *
from util.sudo_labels import *
from archs.model_PALM2_solo import *
from data.bag_loader import *
from data.instance_loader import *
from loss.genSCL import GenSupConLossv2
from loss.palm import PALM
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    


if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "Palm5_OFFICIAL_2"
    
    """dataset_name = 'export_oneLesions' #'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  
    img_size = 300
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  50
    arch = 'resnet50'
    pretrained_arch = False
    """
    
    dataset_name = 'imagenette2_hard'
    label_columns = ['Has_Fish']
    instance_columns = ['Has_Fish']  
    img_size = 128
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    arch = 'efficientnet_b0'
    pretrained_arch = False

    #ITS2CLR Config
    feature_extractor_train_count = 8 # 6
    MIL_train_count = 5
    initial_ratio = .3 #0.3 # --% preditions included
    final_ratio = .8 #0.85 # --% preditions included
    total_epochs = 100
    warmup_epochs = 10
    learning_rate=0.001
    reset_aggregator = False # Reset the model.aggregator weights after contrastive learning
    
    mix_alpha=0.2  #0.2
    mix='mixup'
    
    train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.RandomAffine(degrees=(-90, 90), translate=(0.05, 0.05), scale=(1, 1.2),),
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    val_transform = T.Compose([
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    
    # Get Training Data
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_classes = len(label_columns) + 1
    num_labels = len(label_columns)

    # Create bag datasets
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
    bag_dataset_val = BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False)
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True)


    # Create Model
    model = Embeddingmodel(arch, pretrained_arch, num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")  
    
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 90, lambda_pcon=1).cuda()
    genscl = GenSupConLossv2(temperature=0.07, base_temperature=0.07)
    BCE_loss = nn.BCELoss()
    CE_loss = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
    # MODEL INIT
    config = {
        "head_name": head_name,
        "model_version": model_version,
        "dataset_name": dataset_name,
        "arch": arch,
        "pretrained_arch": pretrained_arch,
        "label_columns": label_columns,
        "instance_columns": instance_columns,
        "img_size": img_size,
        "bag_batch_size": bag_batch_size,
        "min_bag_size": min_bag_size,
        "max_bag_size": max_bag_size,
        "instance_batch_size": instance_batch_size,
        "feature_extractor_train_count": feature_extractor_train_count,
        "MIL_train_count": MIL_train_count,
        "initial_ratio": initial_ratio,
        "final_ratio": final_ratio,
        "total_epochs": total_epochs,
        "reset_aggregator": reset_aggregator,
        "warmup_epochs": warmup_epochs,
        "learning_rate": learning_rate,
    }

    model, optimizer, state = setup_model(model, optimizer, config)
    palm.load_state(state['palm_path'])
    
    # Initialize dictionary for unknown labels
    unknown_labels = {}
    unknown_label_momentum = 0.9
    
    state['pickup_warmup']
    
    # Training loop
    while state['epoch'] < total_epochs:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=state['warmup'], dual_output=True)
            instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True, dual_output=True)
            train_sampler = InstanceSampler(instance_dataset_train, instance_batch_size, strategy=1)
            val_sampler = InstanceSampler(instance_dataset_val, instance_batch_size, strategy=1)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=4, collate_fn = collate_instance, pin_memory=True)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
            if state['warmup']:
                target_count = warmup_epochs
            else:
                target_count = feature_extractor_train_count
            
            
            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            
            for iteration in range(target_count): 
                model.train()
                losses = AverageMeter()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                max_dist = 0
                
                # Iterate over the training data
                for idx, (images, instance_labels, unique_ids) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    im_q, im_k = images
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
                    bsz = instance_labels.shape[0]

                    # forward
                    optimizer.zero_grad()
                    _, _, instance_predictions, features = model(images, pred_on=True, projector=True)
                    features.to(device)
                    
                    # Split features
                    feat_q, feat_k = torch.split(features, [bsz, bsz], dim=0)
                    instance_predictions, _ = torch.split(instance_predictions, [bsz, bsz], dim=0)
        
                    # Create masks for labeled and unlabeled data
                    unlabeled_mask = (instance_labels == -1)
                    labeled_mask = ~unlabeled_mask  # This includes both 0 and 1 labels

                    # Handle labeled instances (0 and 1)
                    if labeled_mask.any():
                        labeled_im_q = im_q[labeled_mask]
                        labeled_im_k = im_k[labeled_mask]
                        labeled_feature_q = feat_q[labeled_mask]
                        labeled_feature_k = feat_k[labeled_mask]
                        labeled_instance_labels = instance_labels[labeled_mask]
                        
                        # GenSCL Loss
                        labeled_im_q, y0a, y0b, lam0 = mix_fn(labeled_im_q, labeled_instance_labels, mix_alpha, mix)
                        labeled_im_k, y1a, y1b, lam1 = mix_fn(labeled_im_k, labeled_instance_labels, mix_alpha, mix)
                        l_q = mix_target(y0a, y0b, lam0, num_classes)
                        l_k = mix_target(y1a, y1b, lam1, num_classes)
                        genscl_loss = genscl([labeled_feature_k, labeled_feature_q], [l_q, l_k], None)
                                        
                        # PALM Loss
                        palm_loss, loss_dict = palm(labeled_feature_k, labeled_instance_labels)
                    else:
                        palm_loss = torch.tensor(0.0).to(device)
                        genscl_loss = torch.tensor(0.0).to(device)

                    combined_labels = instance_labels.clone().float()
    
                    # Handle unlabeled instances (-1)
                    if unlabeled_mask.any():
                        unlabeled_features = feat_q[unlabeled_mask]
                        unlabeled_indices = torch.where(unlabeled_mask)[0]
                        
                        with torch.no_grad():
                            proto_dist, proto_class = palm.get_nearest_prototype(unlabeled_features)
                        
                        for i, idx in enumerate(unlabeled_indices):
                            unique_id = unique_ids[idx]
                            
                            if unique_id not in unknown_labels:
                                unknown_labels[unique_id] = 0.5  # Initialize to 0.5 if not present
                            
                            # Apply momentum update to all instances
                            current_label = unknown_labels[unique_id]
                            new_label = proto_class[i].item()
                            updated_label = unknown_label_momentum * current_label + (1 - unknown_label_momentum) * new_label
                            updated_label = max(0, min(1, updated_label))  # Clamp the updated label to [0, 1]
                            unknown_labels[unique_id] = updated_label
                            
                            # Update the combined_labels tensor with the new pseudo-label
                            combined_labels[idx] = updated_label
                    
                    # Convert confident combined labels to one-hot vectors [Neg class, Pos class]
                    combined_labels_one_hot = torch.zeros(combined_labels.size(0), 2, device=device)
                    combined_labels_one_hot[:, 0] = 1 - combined_labels
                    combined_labels_one_hot[:, 1] = combined_labels

                    # Prepare confident instance predictions
                    instance_predictions_vector = instance_predictions.unsqueeze(1)
                    instance_predictions_vector = torch.cat([1 - instance_predictions_vector, instance_predictions_vector], dim=1)
                    
                    
                    # Calculate CE loss for confident instances
                    ce_loss_value = CE_loss(instance_predictions_vector, combined_labels_one_hot)

                    # Backward pass and optimization step
                    total_loss = palm_loss + ce_loss_value + genscl_loss
                    total_loss.backward()
                    optimizer.step()
        
                    # Update the loss meter
                    losses.update(total_loss.item(), images[0].size(0))
                    
                    # Get predictions from PALM
                    with torch.no_grad():
                        palm_predicted_classes, dist = palm.predict(feat_q)
                        instance_predicted_classes = (instance_predictions) > 0.5

                        # Calculate accuracy for PALM predictions
                        palm_correct = (palm_predicted_classes == instance_labels).sum().item()
                        palm_total_correct += palm_correct
                        
                        # Calculate accuracy for instance predictions
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += instance_labels.size(0)
                        
                        # Update max distance for this epoch
                        max_dist_batch = dist.max().item()
                        max_dist = max(max_dist, max_dist_batch)

                # Calculate accuracies
                palm_train_acc = palm_total_correct / total_samples
                instance_train_acc = instance_total_correct / total_samples
                                
                
                
                # Validation loop
                model.eval()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                val_losses = AverageMeter()

                with torch.no_grad():
                    for idx, (images, instance_labels, _) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        im_q, im_k = images
                        im_q = im_q.cuda(non_blocking=True)
                        im_k = im_k.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)
                        bsz = instance_labels.shape[0]

                        # Forward pass
                        _, _, instance_predictions, features = model(images, pred_on=True, projector=True)
                        features.to(device)

                        # Split features
                        feat_q, feat_k = torch.split(features, [bsz, bsz], dim=0)
                        instance_predictions, _ = torch.split(instance_predictions, [bsz, bsz], dim=0)

                        # GenSCL Loss
                        im_q, y0a, y0b, lam0 = mix_fn(im_q, instance_labels, mix_alpha, mix)
                        im_k, y1a, y1b, lam1 = mix_fn(im_k, instance_labels, mix_alpha, mix)
                        l_q = mix_target(y0a, y0b, lam0, num_classes)
                        l_k = mix_target(y1a, y1b, lam1, num_classes)
                        genscl_loss = genscl([feat_k, feat_q], [l_q, l_k], None)

                        # PALM Loss
                        palm_loss, _ = palm(feat_k, instance_labels, update_prototypes=False)

                        # Convert instance labels to one-hot vectors [Neg class, Pos class]
                        instance_labels_one_hot = torch.zeros(instance_labels.size(0), 2, device=device)
                        instance_labels_one_hot[:, 0] = 1 - instance_labels
                        instance_labels_one_hot[:, 1] = instance_labels

                        # Prepare instance predictions
                        instance_predictions_vector = instance_predictions.unsqueeze(1)
                        instance_predictions_vector = torch.cat([1 - instance_predictions_vector, instance_predictions_vector], dim=1)

                        # Calculate CE loss
                        ce_loss_value = CE_loss(instance_predictions_vector, instance_labels_one_hot)

                        # Calculate total loss
                        total_loss = palm_loss + ce_loss_value + genscl_loss
                        val_losses.update(total_loss.item(), im_q.size(0))

                        # Get predictions
                        palm_predicted_classes, dist = palm.predict(feat_q)
                        instance_predicted_classes = (instance_predictions > 0.5)

                        # Calculate accuracy for PALM predictions
                        palm_correct = (palm_predicted_classes == instance_labels).sum().item()
                        palm_total_correct += palm_correct
                        
                        # Calculate accuracy for instance predictions
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += instance_labels.size(0)

                # Calculate accuracies
                palm_val_acc = palm_total_correct / total_samples if total_samples > 0 else 0
                instance_val_acc = instance_total_correct / total_samples if total_samples > 0 else 0

                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train Palm Acc: {palm_train_acc:.5f}, Train FC Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss: {val_losses.avg:.5f}, Val Palm Acc: {palm_val_acc:.5f}, Val FC Acc: {instance_val_acc:.5f}')
                
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
                        save_state(state['epoch'], label_columns, instance_train_acc, val_losses.avg, instance_val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                        palm.save_state(os.path.join(target_folder, "palm_state.pkl"), max_dist)
                        print("Saved checkpoint due to improved val_loss_instance")



        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            

        
            
        print('\nTraining Bag Aggregator')
        for iteration in range(MIL_train_count):
            model.train()
            train_bag_logits = {}
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0

            for (images, yb, instance_labels, id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
                num_bags = len(images)
                optimizer.zero_grad()

                # Forward pass
                bag_pred, _, instance_pred, features = model(images, pred_on=True)
                

                # Split the embeddings back into per-bag embeddings
                split_sizes = [bag.size(0) for bag in images]
                y_hat_per_bag = torch.split(instance_pred, split_sizes, dim=0)
                for i, y_h in enumerate(y_hat_per_bag):
                    train_bag_logits[id[i].item()] = y_h.detach().cpu().numpy()
                
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
                    bag_pred, _, _, features = model(images, pred_on=True)

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
            
            print(f"[{iteration+1}/{MIL_train_count}] | Acc | Loss")
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
                
                save_state(state['epoch'], label_columns, train_acc, val_loss, val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                palm.save_state(os.path.join(target_folder, "palm_state.pkl"), max_dist)
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1
                


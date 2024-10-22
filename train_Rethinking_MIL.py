import os, pickle
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from archs.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from util.format_data import *
from util.sudo_labels import *
from archs.model_INS import *
from data.bag_loader import *
from data.instance_loader import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class IWSCL(nn.Module):
    def __init__(self, feat_dim, num_classes=2, momentum=0.999, temperature=0.07):
        super(IWSCL, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.momentum = momentum
        self.num_classes = num_classes
        self.temperature = temperature

    def forward(self, features, instance_predictions, instance_labels, queue, queue_labels, val_on=False):
        """
        Args:
            features: Current batch instance features [B, feat_dim]
            instance_predictions: Predicted labels from classifier [B, num_classes]
            instance_labels: Ground truths, -1 if unknown
            queue: Feature queue [Q, feat_dim]
            queue_labels: Labels for queue features [Q, num_classes]
            val_on: Whether in validation mode
        """
 
        # Fix queue shape
        queue = queue.T

        # Compute pairwise distances
        distances = torch.cdist(features, features, p=2)  # Euclidean distance matrix
        
        # Create masks for positive and negative pairs
        labels_matrix = instance_labels.expand(len(instance_labels), len(instance_labels))
        positive_mask = labels_matrix.eq(labels_matrix.T)
        negative_mask = ~positive_mask
        
        # Remove diagonal elements (self-similarity)
        mask_no_diagonal = ~torch.eye(len(instance_labels), dtype=torch.bool, device=features.device)
        positive_mask = positive_mask & mask_no_diagonal
        negative_mask = negative_mask & mask_no_diagonal
        
        # Positive loss: pull same-class samples together
        positive_loss = (distances * positive_mask).sum() / positive_mask.sum()
        
        # Negative loss: push different-class samples apart
        # Use max(0, margin - distance) to create a margin between classes
        margin = 2.0
        negative_loss = torch.clamp(margin - distances, min=0.0)
        negative_loss = (negative_loss * negative_mask).sum() / negative_mask.sum()
        
        # Total loss
        loss = positive_loss + negative_loss



        # Update prototypes
        with torch.no_grad():
            for c in range(self.num_classes):
                class_features = features[instance_predictions == c]
                if class_features.size(0) > 0:
                    new_prototype = class_features.mean(dim=0)
                    self.prototypes.data[c] = self.momentum * self.prototypes.data[c] + (1 - self.momentum) * new_prototype
                    self.prototypes.data[c] = F.normalize(self.prototypes.data[c], dim=0)
        
        # Generate pseudo labels
        similarities = torch.matmul(features, self.prototypes.t())
        pseudo_labels = similarities.argmax(dim=1)
        
        # Override pseudo labels with ground truth when available
        ground_truth_mask = instance_labels != -1
        pseudo_labels[ground_truth_mask] = instance_labels[ground_truth_mask]

        return loss, pseudo_labels
        
        
if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "TEST_R_MIL"
    
    """dataset_name = 'export_oneLesions' #'export_03_18_2024' or 'export_oneLesions'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  
    img_size = 224
    bag_batch_size = 3
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  50
    arch = 'efficientnet_b0'
    pretrained_arch = False"""

    
    dataset_name = 'imagenette2_hard2'
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
    bags_train, bags_val = prepare_all_data(config)
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
    
    # LOSS INIT
    BCE_loss = nn.BCEWithLogitsLoss()
    CE_crit = nn.CrossEntropyLoss()
    IWSCL_crit = IWSCL(128).to(device)
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001
    
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, optimizer, config)

    
    # Training loop
    while state['epoch'] < total_epochs:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=state['warmup'], dual_output=True)
            instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True, dual_output=True)
            train_sampler = InstanceSampler(instance_dataset_train, instance_batch_size, strategy=1)
            val_sampler = InstanceSampler(instance_dataset_val, instance_batch_size, strategy=1)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=2, collate_fn = collate_instance, pin_memory=True)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
            if state['warmup']:
                target_count = warmup_epochs
            else:
                target_count = feature_extractor_train_count
            
            
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')

            
            
            for iteration in range(target_count): 
                losses = AverageMeter()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                model.train()
                
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
                    losses.update(total_loss.item(), im_q[0].size(0))
                    
                    # Get predictions
                    with torch.no_grad():
                        instance_predicted_classes = (instance_predictions) > 0.5
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        total_samples += instance_labels.size(0)

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
                        total_loss = ce_loss + 0
                        val_losses.update(total_loss.item(), im_q[0].size(0))

                        # Get predictions
                        instance_predicted_classes = (instance_predictions) > 0.5
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        total_samples += instance_labels.size(0)

                # Calculate accuracies
                palm_val_acc = palm_total_correct / total_samples
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
                        save_state(state['epoch'], label_columns, instance_train_acc, val_losses.avg, instance_val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
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
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1
                
                # Create selection mask
                """predictions_ratio = prediction_anchor_scheduler(state['epoch'], total_epochs, 0, initial_ratio, final_ratio)
                state['selection_mask'] = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(state['selection_mask'], file)"""


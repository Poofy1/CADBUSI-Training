import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
import pickle
from torch import nn
from archs.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from torch.utils.data import Sampler
from util.format_data import *
from util.sudo_labels import *
from archs.model_PALM2_solo import *
from loss.palm import PALM
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=True):
        self.transform = transform
        self.warmup = warmup

        self.images = []
        self.final_labels = []

        
        for bag_id, bag_info in bags_dict.items():
            images = bag_info['images']
            image_labels = bag_info['image_labels']
            bag_label = bag_info['bag_labels'][0]  # Assuming each bag has a single label
            
            bag_id_key = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
            
            
            if bag_id_key in selection_mask:
                selection_mask_labels, _ = selection_mask[bag_id_key]
            else: 
                selection_mask_labels = None

            for idx, (img, label) in enumerate(zip(images, image_labels)):
                image_label = None
                
                if self.warmup:
                    # Only include confident instances (selection_mask) or negative bags or instance labels
                    if label[0] is not None:
                        image_label = label[0]
                    elif bag_label == 0:
                        image_label = 0
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                else:
                    # Return all images with unknown possiblity 
                    if label[0] is not None:
                        image_label = label[0]
                    elif bag_label == 0:
                        image_label = 0
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                    else:
                        image_label = -1
                
                if image_label is not None:
                    self.images.append(img)
                    self.final_labels.append(image_label)

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        
        img = Image.open(img_path).convert("RGB")
        image_data_q = self.transform(img)
        image_data_k = self.transform(img)

        return (image_data_q, image_data_k), instance_label
    
    def __len__(self):
        return len(self.images)


class WarmupSampler(Sampler):
    def __init__(self, dataset, batch_size, strategy=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.indices_0 = [i for i, label in enumerate(self.dataset.final_labels) if label == 0]
        self.indices_1 = [i for i, label in enumerate(self.dataset.final_labels) if label == 1]

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size

        for _ in range(total_batches):
            if self.strategy == 1:
                # Ensure at least one positive sample
                num_positives = random.randint(1, max(1, min(len(self.indices_1), self.batch_size - 1)))
                num_negatives = self.batch_size - num_positives
            elif self.strategy == 2:
                # Aim for 50/50 balance
                num_positives = self.batch_size // 2
                num_negatives = self.batch_size - num_positives
            else:
                raise ValueError("Invalid strategy. Choose 1 or 2")

            batch_positives = random.sample(self.indices_1, num_positives)
            batch_negatives = random.sample(self.indices_0, num_negatives)

            batch = batch_positives + batch_negatives
            random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    
def collate_instance(batch):
    batch_data_q = []
    batch_data_k = [] 
    batch_labels = []

    for (image_data_q, image_data_k), bag_label in batch:
        batch_data_q.append(image_data_q)
        batch_data_k.append(image_data_k)
        batch_labels.append(bag_label)

    # Stack the images and labels
    batch_data_q = torch.stack(batch_data_q)
    batch_data_k = torch.stack(batch_data_k)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return (batch_data_q, batch_data_k), batch_labels



class GenSupConLossv2(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(GenSupConLossv2, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels, anc_mask = None):
        '''
        Args:
            feats: (anchor_features, contrast_features), each: [N, feat_dim]
            labels: (anchor_labels, contrast_labels) each: [N, num_cls]
            anc_mask: (anchors_mask, contrast_mask) each: [N]
        '''

        anchor_labels = torch.cat(labels, dim=0).float()
        contrast_labels = anchor_labels
        anchor_features = torch.cat(features, dim=0)
        contrast_features = anchor_features
        
        # 1. compute similarities among targets
        anchor_norm = torch.norm(anchor_labels, p=2, dim=-1, keepdim=True) # [anchor_N, 1]
        contrast_norm = torch.norm(contrast_labels, p=2, dim=-1, keepdim=True) # [contrast_N, 1]
        deno = torch.mm(anchor_norm, contrast_norm.T)
        mask = torch.mm(anchor_labels, contrast_labels.T) / deno # cosine similarity: [anchor_N, contrast_N]
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        # 2. compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_features, contrast_features.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        if anc_mask:
            select = torch.cat( anc_mask )
            loss = loss[select]
            
        loss = loss.mean()

        return loss

    


    



if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "Palm4_test41234"
    
    """dataset_name = 'export_oneLesions' #'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  
    img_size = 224
    bag_batch_size = 3
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    arch = 'efficientnet_b0'
    pretrained_arch = False"""

    
    dataset_name = 'imagenette2_hard'
    label_columns = ['Has_Fish']
    instance_columns = ['Has_Fish']  
    img_size = 128
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    arch = 'resnet18'
    pretrained_arch = False

    #ITS2CLR Config
    feature_extractor_train_count = 6 # 6
    MIL_train_count = 6
    initial_ratio = .3 #0.3 # --% preditions included
    final_ratio = .8 #0.85 # --% preditions included
    total_epochs = 9999
    warmup_epochs = 20
    learning_rate=0.001
    reset_aggregator = True # Reset the model.aggregator weights after contrastive learning

    mix_alpha=0  #0.2
    mix='mixup'
    
    
    train_transform = T.Compose([
                ###T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.05), scale=(1, 1.2),),
                CLAHETransform(),
                T.ToTensor(),
                ###GaussianNoise(mean=0, std=0.015), 
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
    
    
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 5, lambda_pcon=1).cuda() #lambda_pcon = 0 means prototypes are not moved
    genscl = GenSupConLossv2(temperature=0.07, base_temperature=0.07)
    BCE_loss = nn.BCELoss()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
    
    
    
    
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
    
    # Training loop
    while state['epoch'] < total_epochs:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=True)
            instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True)
            train_sampler = WarmupSampler(instance_dataset_train, instance_batch_size, strategy=1)
            val_sampler = WarmupSampler(instance_dataset_val, instance_batch_size, strategy=1)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=4, collate_fn = collate_instance, pin_memory=True)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
            if state['warmup']:
                target_count = warmup_epochs
            else:
                target_count = feature_extractor_train_count
            
            
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            
            # Unfreeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = True

            
            
            for i in range(target_count): 
                model.train()
                losses = AverageMeter()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                
                # Iterate over the training data
                for idx, (images, instance_labels) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    bsz = instance_labels.shape[0]
                    im_q, im_k = images
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
                    
                    # forward
                    optimizer.zero_grad()
                    _, _, instance_predictions, features = model(images, pred_on=True, projector=True)
                    features.to(device)
                    
                    

                    
                    # image-based regularizations (lam 1 = no mixup)
                    im_q, y0a, y0b, lam0 = mix_fn(im_q, instance_labels, mix_alpha, mix)
                    im_k, y1a, y1b, lam1 = mix_fn(im_k, instance_labels, mix_alpha, mix)
                    images = [im_q, im_k]
                    l_q = mix_target(y0a, y0b, lam0, num_classes)
                    l_k = mix_target(y1a, y1b, lam1, num_classes)

                    zk, zq = torch.split(features, [bsz, bsz], dim=0)
                    
                    # Get loss contrastive loss
                    palm_loss, loss_dict = palm(zk, instance_labels)
                    
                    # get loss (no teacher)
                    genscl_loss = genscl([zk, zq], [l_q, l_k], None)
                    #print(genscl_loss)
                    
                    # Calculate BCE loss
                    bce_loss_value = BCE_loss(instance_predictions[:bsz], instance_labels.float())

                    # Backward pass and optimization step
                    total_loss = palm_loss + bce_loss_value + genscl_loss
                    total_loss.backward()
                    optimizer.step()
        
                    # Update the loss meter
                    losses.update(total_loss.item(), images[0].size(0))
                    
                    # Get predictions from PALM
                    with torch.no_grad():
                        palm_predicted_classes = palm.predict(zk)
                        instance_predicted_classes = (instance_predictions[:bsz]) > 0.5

                    # Calculate accuracy for PALM predictions
                    palm_correct = (palm_predicted_classes == instance_labels).sum().item()
                    palm_total_correct += palm_correct
                    
                    # Calculate accuracy for instance predictions
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

                with torch.no_grad():
                    for idx, (images, instance_labels) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        im_q, im_k = images
                        im_q = im_q.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)

                        # Forward pass
                        _, _, instance_predictions, features = model(im_q, projector=True)
                        features.to(device)

                        # Get predictions
                        palm_predicted_classes = palm.predict(features)
                        instance_predicted_classes = (instance_predictions) > 0.5

                        # Calculate accuracy for PALM predictions
                        palm_correct = (palm_predicted_classes == instance_labels).sum().item()
                        palm_total_correct += palm_correct
                        
                        # Calculate accuracy for instance predictions
                        instance_correct = (instance_predicted_classes == instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += instance_labels.size(0)

                # Calculate accuracies
                palm_val_acc = palm_total_correct / total_samples
                instance_val_acc = instance_total_correct / total_samples
                
                print(f'[{i+1}/{target_count}] Train Loss: {losses.avg:.5f}, Train Palm Acc: {palm_train_acc:.5f}, Train FC Acc: {instance_train_acc:.5f}')
                print(f'[{i+1}/{target_count}] Val Loss:   N/A, Val Palm Acc: {palm_val_acc:.5f}, Val FC Acc: {instance_val_acc:.5f}')
                
                # Save the model
                if instance_val_acc > state['val_acc_best']:
                    state['val_acc_best'] = instance_val_acc
                    if state['warmup']:
                        target_folder = state['head_folder']
                        target_name = state['pretrained_name']
                    else:
                        target_folder = state['model_folder']
                        target_name = state['model_name']
                    all_targs = []
                    all_preds = []
                    
                    
                    save_state(state['epoch'], label_columns, instance_train_acc, 0, instance_val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                    palm.save_state(os.path.join(target_folder, "palm_state.pkl"))
                    print("Saved checkpoint due to improved val_acc")




        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
            

        
            
        print('Predicting Missing Instances')
        for a in range(MIL_train_count):
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
            
            print(f"[{a+1}/{MIL_train_count}] | Acc | Loss")
            print(f"Train | {train_acc:.4f} | {train_loss:.4f}")
            print(f"Val | {val_acc:.4f} | {val_loss:.4f}")

            # Save the model
            if val_loss < state['val_loss_best']:
                state['val_loss_best'] = val_loss
                if state['warmup']:
                    target_folder = state['head_folder']
                    target_name = state['pretrained_name']
                else:
                    target_folder = state['model_folder']
                    target_name = state['model_name']
                
                save_state(state['epoch'], label_columns, train_acc, val_loss, val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                print("Saved checkpoint due to improved val_loss")

                
                state['epoch'] += 1
                
                # Create selection mask
                predictions_ratio = prediction_anchor_scheduler(state['epoch'], total_epochs, 0, initial_ratio, final_ratio)
                selection_mask = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(selection_mask, file)


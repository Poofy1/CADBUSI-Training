import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
import pickle
from torch import nn
from archs.save_arch import *
from data.Gen_ITS2CLR_util import *
from torch.utils.data import Sampler
from torch.optim import Adam
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from data.format_data import *
from archs.model_GenSCL_DSMIL import *
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
        self.warmup_mask = []
        
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
                warmup_mask_value = 0
                
                if not self.warmup:
                    # Only include confident instances (selection_mask) or negative bags or instance labels
                    if label[0] is not None:
                        image_label = label[0]
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                    elif bag_label == 0:
                        image_label = 0
                else:
                    # Include all data but replace with image_labels if present
                    if label[0] is not None:
                        image_label = label[0]
                    else:
                        image_label = bag_label  # Use bag label if instance label is not present
                        if bag_label == 1:
                            warmup_mask_value = 1 # Set warmup mask to 1 for instances without label[0] and bag_label is 1
                
                if image_label is not None:
                    self.images.append(img)
                    self.final_labels.append(image_label)
                    self.warmup_mask.append(warmup_mask_value)

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        warmup_unconfident = self.warmup_mask[index]
        
        img = Image.open(img_path).convert("RGB")
        image_data_q = self.transform(img)
        image_data_k = self.transform(img)

        return (image_data_q, image_data_k), instance_label, warmup_unconfident


    def __len__(self):
        return len(self.images)


class WarmupSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_0 = [i for i, mask in enumerate(self.dataset.warmup_mask) if mask == 0]
        self.indices_1 = [i for i, mask in enumerate(self.dataset.warmup_mask) if mask == 1]

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size

        for _ in range(total_batches):
            # Randomly decide how many mask 0s to include, ensuring at least 1
            num_mask_0 = random.randint(1, max(1, min(len(self.indices_0), self.batch_size - 1)))
            batch_mask_0 = random.sample(self.indices_0, num_mask_0)

            # Fill the rest of the batch with mask 1s, if any space left
            num_mask_1 = self.batch_size - num_mask_0
            batch_mask_1 = random.sample(self.indices_1, num_mask_1) if num_mask_1 > 0 else []

            batch = batch_mask_0 + batch_mask_1
            random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
def collate_instance(batch):
    batch_data_q = []
    batch_data_k = [] 
    batch_labels = []
    batch_unconfident = []

    for (image_data_q, image_data_k), bag_label, warmup_unconfident in batch:
        batch_data_q.append(image_data_q)
        batch_data_k.append(image_data_k)
        batch_labels.append(bag_label)
        batch_unconfident.append(warmup_unconfident)

    # Stack the images and labels
    batch_data_q = torch.stack(batch_data_q)
    batch_data_k = torch.stack(batch_data_k)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    batch_unconfident = torch.tensor(batch_unconfident, dtype=torch.long)

    return (batch_data_q, batch_data_k), batch_labels, batch_unconfident


def collate_bag(batch):
    batch_data = []
    batch_bag_labels = []
    batch_instance_labels = []
    batch_ids = []

    for sample in batch:
        image_data, bag_labels, instance_labels, bag_id = sample  # Updated to unpack four items
        batch_data.append(image_data)
        batch_bag_labels.append(bag_labels)
        batch_instance_labels.append(instance_labels)
        batch_ids.append(bag_id)

    # Use torch.stack for bag labels to handle multiple labels per bag
    out_bag_labels = torch.stack(batch_bag_labels).cuda()

    # Converting to a tensor
    out_ids = torch.tensor(batch_ids, dtype=torch.long).cuda()

    return batch_data, out_bag_labels, batch_instance_labels, out_ids



class GenSupConLossv2(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(GenSupConLossv2, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, anc_mask = None):
        '''
        Args:
            feats: (anchor_features, contrast_features), each: [N, feat_dim]
            labels: (anchor_labels, contrast_labels) each: [N, num_cls]
            anc_mask: (anchors_mask, contrast_mask) each: [N]
        '''
        if self.contrast_mode == 'all': # anchor+contrast @ anchor+contrast
            anchor_labels = torch.cat(labels, dim=0).float()
            contrast_labels = anchor_labels
            
            anchor_features = torch.cat(features, dim=0)
            contrast_features = anchor_features
        elif self.contrast_mode == 'one': # anchor @ contrast
            anchor_labels = labels[0].float()
            contrast_labels = labels[1].float()
            
            anchor_features = features[0]
            contrast_features = features[1]
            
        # 1. compute similarities among targets
        anchor_norm = torch.norm(anchor_labels, p=2, dim=-1, keepdim=True) # [anchor_N, 1]
        contrast_norm = torch.norm(contrast_labels, p=2, dim=-1, keepdim=True) # [contrast_N, 1]
        
        deno = torch.mm(anchor_norm, contrast_norm.T)
        mask = torch.mm(anchor_labels, contrast_labels.T) / deno # cosine similarity: [anchor_N, contrast_N]
        
        logits_mask = torch.ones_like(mask)
        if self.contrast_mode == 'all':
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
    
    


def create_selection_mask(train_bag_logits, include_ratio):
    combined_probs = []
    original_indices = []
    predictions = []
    
    # Loop through train_bag_logits to process probabilities
    for bag_id, probs in train_bag_logits.items():
        # Convert tensor bag_id to integer if necessary
        bag_id_int = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
        for i, prob in enumerate(probs):
            combined_probs.append(prob.item())
            original_indices.append((bag_id_int, i))
            predictions.append(prob.item())

    total_predictions = len(combined_probs)
    predictions_included = int(total_predictions * include_ratio)
    print(f'Including Predictions: {include_ratio:.2f} ({predictions_included})')

    # Rank instances based on their confidence (distance from 0.5)
    confidence_scores = np.abs(np.array(combined_probs) - 0.5)
    top_indices = np.argsort(-confidence_scores)[:predictions_included]

    # Initialize combined_dict with all -1 for masks (not selected by default) and placeholders for predictions
    combined_dict = {}
    for bag_id, probs in train_bag_logits.items():
        bag_id_int = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
        mask = np.full(len(probs), -1, dtype=int)  # Initialize mask to -1 (not selected)
        pred_list = [None] * len(probs)  # Initialize prediction list
        combined_dict[bag_id_int] = [mask, pred_list]

    # Update predictions in combined_dict for all instances
    for idx, (bag_id_int, pos) in enumerate(original_indices):
        combined_dict[bag_id_int][1][pos] = predictions[idx]  # Update prediction

    # Set mask based on selection
    for idx in top_indices:
        original_bag_id, original_position = original_indices[idx]
        prob = combined_probs[idx]
        # Update mask based on probability: 0 if below 0.5, 1 if above 0.5
        combined_dict[original_bag_id][0][original_position] = int(prob > 0.5)

    return combined_dict
    
    

def load_state(stats_path, target_folder):
    selection_mask = []
    
    with open(stats_path, 'rb') as f:
        saved_stats = pickle.load(f)
        train_losses = saved_stats['train_losses']
        valid_losses = saved_stats['valid_losses']
        epoch = saved_stats['epoch']
        val_loss_best = saved_stats['val_loss']
    
    # Load the selection_mask dictionary from the file
    if os.path.exists(f'{target_folder}/selection_mask.pkl'):
        with open(f'{target_folder}/selection_mask.pkl', 'rb') as file:
            selection_mask = pickle.load(file)
            
    return train_losses, valid_losses, epoch, val_loss_best, selection_mask


if __name__ == '__main__':

    # Config
    model_version = '03'
    dataset_name = 'cifar10'
    
    label_columns = ['Has_Truck']
    instance_columns = ['']
    img_size = 32
    bag_batch_size = 30
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  200
    
    """pretrained_name = "03_18_2024_Res50_Head_5"
    model_name = '03_18_2024_Res50_05'
    dataset_name = 'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  
    img_size = 300
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25"""
    use_efficient_net = False
    
    #ITS2CLR Config
    feature_extractor_train_count = 6
    MIL_train_count = 8
    initial_ratio = 1 #0.3 # --% preditions included
    final_ratio = 1 #0.85 # --% preditions included
    total_epochs = 20
    warmup_epochs = 15
    
    arch = "resnet50"
    pretrained_arch = False
    reset_aggregator = True # Reset the model.aggregator weights after contrastive learning
    
    learning_rate=0.001
    mix_alpha=0.2
    mix='mixup'
    num_classes = len(label_columns) + 1

    
    # Get Training Data
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_labels = len(label_columns)
    
    train_transform = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    val_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    """train_transform = T.Compose([
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
            ])"""


    # Create datasets
    #bag_dataset_train = TUD.Subset(BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False),list(range(0,100)))
    #bag_dataset_val = TUD.Subset(BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False),list(range(0,100)))
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=True)
    bag_dataset_val = BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False)
     
    # Create bag data loaders
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True)


    # Get Model
    if "resnet" in arch:
        encoder = create_timm_body(arch, pretrained=pretrained_arch)
        nf = num_features_model( nn.Sequential(*encoder.children()))
    else:
        encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        nf = 512
        # Replace the last fully connected layer with a new one
        num_features = encoder.classifier[1].in_features
        encoder.classifier[1] = nn.Linear(num_features, nf)
    

    model = Embeddingmodel(encoder = encoder, nf = nf, num_classes = num_labels, efficient_net = use_efficient_net).cuda()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")        
        
    optimizer = Adam(model.parameters(), lr=learning_rate)
    BCE_loss = nn.BCELoss()
    genscl = GenSupConLossv2(temperature=0.07, contrast_mode='all', base_temperature=0.07)
    train_losses = []
    valid_losses = []

    model_name = f"{dataset_name}_{arch}_{model_version}"
    pretrained_name = f"Head_{model_name}"
    
    
    # Check if the model already exists
    model_folder = f"{env}/models/{model_name}/"
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"
    
    head_folder = f"{env}/models/{pretrained_name}/"
    head_path = f"{head_folder}/{pretrained_name}.pth"
    head_optimizer_path = f"{head_folder}/{pretrained_name}_optimizer.pth"
    
    val_loss_best = 99999
    selection_mask = []
    epoch = 0
    warmup = False
    pickup_warmup = False

    if os.path.exists(model_path): # If main model exists
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"Loaded pre-existing model from {model_name}")

        train_losses, valid_losses, epoch, val_loss_best, selection_mask = load_state(stats_path, model_folder)
        
    else:
        print(f"{model_name} does not exist, creating new instance")
        os.makedirs(model_folder, exist_ok=True)
        
        # Save the current configuration as a human-readable file
        config = {
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
            "use_efficient_net": use_efficient_net,
            "feature_extractor_train_count": feature_extractor_train_count,
            "MIL_train_count": MIL_train_count,
            "initial_ratio": initial_ratio,
            "final_ratio": final_ratio,
            "total_epochs": total_epochs,
            "reset_aggregator": reset_aggregator,
            "warmup_epochs": warmup_epochs,
            "learning_rate": learning_rate,
            "mix_alpha": mix_alpha
        }
        

        if os.path.exists(head_path):  # If main head model exists
            pickup_warmup = True
            #model.load_state_dict(torch.load(head_path))
            
            # Load only the encoder state dictionary
            encoder_state_dict = torch.load(head_path)
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items() if k.startswith('encoder.')}
            model.encoder.load_state_dict(encoder_state_dict)
            optimizer.load_state_dict(torch.load(head_optimizer_path))
            print(f"Loaded pre-trained model from {pretrained_name}")
            
            config_path = f"{model_folder}/config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
                
            with open(f'{model_folder}model_architecture.txt', 'w') as f:
                print(model, file=f)
             
        else: # If main head model DOES NOT exists
            warmup = True
            os.makedirs(head_folder, exist_ok=True)

                
            config_path = f"{head_folder}/config.txt"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            with open(f'{head_folder}model_architecture.txt', 'w') as f:
                print(model, file=f)
            

    # Training loop
    while epoch < total_epochs:
        
        print(f'Warmup Mode: {warmup}')
        
        if not pickup_warmup: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, selection_mask, transform=train_transform, warmup=warmup)
            
            if warmup:
                sampler = WarmupSampler(instance_dataset_train, instance_batch_size)
                instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=sampler, collate_fn = collate_instance)
                target_count = warmup_epochs
            else:
                instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_size=instance_batch_size, collate_fn = collate_instance, drop_last=True, shuffle = True)
                target_count = feature_extractor_train_count
            

            print('Training Feature Extractor')
            
            # Unfreeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = True
        
            # Generalized Supervised Contrastive Learning phase
            
            model.train()
            for i in range(target_count): 
                losses = AverageMeter()

                # Iterate over the training data
                for idx, (images, instance_labels, unconfident_mask) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
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
                    features = F.normalize(features, dim=1)
                    zk, zq = torch.split(features, [bsz, bsz], dim=0)
                    
                    # get loss (no teacher)
                    mapped_anchors = ~unconfident_mask.bool()
                    loss = genscl([zk, zq], [l_q, l_k], (mapped_anchors, mapped_anchors))
                    losses.update(loss.item(), bsz)

                    loss.backward()
                    optimizer.step()
                    
                print(f'[{i+1}/{target_count}] Gen_SCL Loss: {losses.avg:.5f}')



        if pickup_warmup: 
            pickup_warmup = False
        if warmup:
            # Save the model and optimizer
            torch.save(model.state_dict(), head_path)
            torch.save(optimizer.state_dict(), head_optimizer_path)
            
            print("Warmup Phase Finished")
            warmup = False
            
            

        print('Training Aggregator')
        
        if reset_aggregator:
            model.aggregator.reset_parameters() # Reset the model.aggregator weights before training
        
        # Freeze the encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        # Training phase
        for i in range(MIL_train_count):
            
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

            train_losses.append(train_loss)
            valid_losses.append(val_loss)

            print(f"[{i+1}/{MIL_train_count}] | Acc | Loss")
            print(f"Train | {train_acc:.4f} | {train_loss:.4f}")
            print(f"Val | {val_acc:.4f} | {val_loss:.4f}")
                        
            
            
            

            # Save the model
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                if warmup:
                    target_folder = head_folder
                    target_name = pretrained_name
                else:
                    target_folder = model_folder
                    target_name = model_name
                
                save_state(epoch, label_columns, train_acc, val_loss, val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, train_losses, valid_losses,)
                print("Saved checkpoint due to improved val_loss")
                
                # Create selection mask
                predictions_ratio = prediction_anchor_scheduler(epoch, total_epochs, 0, initial_ratio, final_ratio)
                #predictions_ratio = .9
                selection_mask = create_selection_mask(train_bag_logits, predictions_ratio)
                print("Created new sudo labels")
                
                epoch += 1
                
                # Save selection
                with open(f'{target_folder}/selection_mask.pkl', 'wb') as file:
                    pickle.dump(selection_mask, file)

            
            

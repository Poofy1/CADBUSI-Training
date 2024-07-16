import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from archs.save_arch import *
from torch.optim import Adam
from data.format_data import *
from archs.model_ABMIL import *
from archs.backbone import create_timm_body
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    
    
class ITS2CLR_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, train=True, save_processed=False, bag_type='all'):
        self.bags_dict = bags_dict
        self.save_processed = save_processed
        self.train = train
        self.unique_bag_ids = list(bags_dict.keys())

        # Filter bags based on bag_type
        if bag_type != 'all':
            self.unique_bag_ids = [bag_id for bag_id in self.unique_bag_ids
                                if (self.bags_dict[bag_id]['bag_labels'][0] == 1 and bag_type == 'positive') or 
                                    (self.bags_dict[bag_id]['bag_labels'][0] == 0 and bag_type == 'negative')]
            
        # Normalize
        if train:
            self.tsfms = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.05), scale=(1, 1.2),),
                #HistogramEqualization(),
                CLAHETransform(),
                T.ToTensor(),
                GaussianNoise(mean=0, std=0.015),  # Add slight noise
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tsfms = T.Compose([
                #HistogramEqualization(),
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, index):
        actual_id = self.unique_bag_ids[index]
        bag_info = self.bags_dict[actual_id]

        # Extract labels, image file paths, and instance-level labels
        bag_labels = bag_info['bag_labels']
        files_this_bag = bag_info['images']
        instance_labels = bag_info['image_labels']

        # Process images
        image_data = torch.stack([self.tsfms(Image.open(fn).convert("RGB")) for fn in files_this_bag])
        image_data = image_data.cuda()  # Move to GPU if CUDA is available
        
        # Save processed images if required
        if self.save_processed:
            save_folder = os.path.join(env, 'processed_images')  
            os.makedirs(save_folder, exist_ok=True)
            for idx, img_tensor in enumerate(image_data):
                img_save_path = os.path.join(save_folder, f'bag_{actual_id}_img_{idx}.png')
                img_tensor = unnormalize(img_tensor)
                img = TF.to_pil_image(img_tensor.cpu().detach())
                img.save(img_save_path)

        # Convert bag labels list to a tensor
        bag_labels_tensor = torch.tensor(bag_labels, dtype=torch.float32)

        # Convert instance labels to a tensor, using -1 for None
        instance_labels_tensors = [torch.tensor(labels, dtype=torch.float32) if labels != [None] else torch.tensor([-1], dtype=torch.float32) for labels in instance_labels]

        return image_data, bag_labels_tensor, instance_labels_tensors, actual_id

    
    def __len__(self):
        return len(self.unique_bag_ids)
    
    def n_features(self):
        return self.data.size(1)
    

class EmbeddingBagModel(nn.Module):
    
    def __init__(self, encoder, aggregator, num_classes=1):
        super(EmbeddingBagModel, self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.num_classes = num_classes
                    
    def forward(self, input):
        num_bags = len(input) # input = [bag #, image #, channel, height, width]
        
        # Concatenate all bags into a single tensor for batch processing
        all_images = torch.cat(input, dim=0)  # Shape: [Total images in all bags, channel, height, width]
        
        # Calculate the embeddings for all images in one go
        h_all = self.encoder(all_images)
        
        # Split the embeddings back into per-bag embeddings
        split_sizes = [bag.size(0) for bag in input]
        h_per_bag = torch.split(h_all, split_sizes, dim=0)
        logits = torch.empty(num_bags, self.num_classes).cuda()
        saliency_maps, yhat_instances, attention_scores = [], [], []
        
        for i, h in enumerate(h_per_bag):
            # Receive four values from the aggregator
            yhat_bag, sm, yhat_ins, att_sc = self.aggregator(h)
            
            logits[i] = yhat_bag
            saliency_maps.append(sm)
            yhat_instances.append(yhat_ins)
            attention_scores.append(att_sc)
        
        return logits, saliency_maps, yhat_instances, attention_scores, h_all


def generate_pseudo_labels(inputs, threshold = .5):
    pseudo_labels = []
    for tensor in inputs:
        # Calculate the dynamic threshold for each tensor
        threshold = 1 / tensor.size(0)
        pseudo_labels_tensor = (tensor > threshold).float()
        pseudo_labels.append(pseudo_labels_tensor)
    
    #print("input scores: ", inputs)    
    #print("pseudo labels: ", pseudo_labels)
    return pseudo_labels


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='one',
                 base_temperature=0.07, pair_mode=0, mask_uncertain_neg=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.pair_mode = pair_mode
        self.mask_uncertain_neg = mask_uncertain_neg

    def forward(self, features, labels=None, bag_label=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            if self.pair_mode==0:
                pass

            # only take the positive into account
            elif self.pair_mode==1:
                mask[labels[:, 0] == 0, :] = 0
                mask[:, labels[:, 0] == 0] = 0

            # only take the negative into account
            elif self.pair_mode==2:
                mask[labels[:, 0] == 1, :] = 0
                mask[:, labels[:, 0] == 1] = 0
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        if self.mask_uncertain_neg:
            # to mask out the instance with bag-label being 1 and instance-label being 0
            labels_repeat = labels.repeat(anchor_count, 1)
            bag_label_repeat = bag_label.repeat(anchor_count, 1)

            labels_repeat = labels_repeat.to(device)
            bag_label_repeat = bag_label_repeat.to(device)

            logits_mask[:, (bag_label_repeat[:, 0] == 1) * (labels_repeat[:, 0] == 0)] = 0


        mask = mask * logits_mask
    
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        if self.mask_uncertain_neg:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+ 1e-10)
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) )

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            
        
        loss = loss.view(anchor_count, batch_size).mean()

        #print("Bag Loss: ", loss)
        #print("anchor_count: ", anchor_count)
        #print("batch_size: ", batch_size)

        return loss

def spl_scheduler(current_epoch, total_epochs, initial_ratio, final_ratio):
    if current_epoch < warmup_epochs:
        return initial_ratio
    else:
        return initial_ratio + (final_ratio - initial_ratio) * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)

def combine_pos_neg_data(pos_data, neg_data, pos_batch_size, neg_batch_size):
    if pos_batch_size == 0:
        return neg_data
    elif neg_batch_size == 0:
        return pos_data
    
    # Unpack data from positive and negative batches
    pos_xb, pos_yb, pos_instance_yb, pos_id = pos_data
    neg_xb, neg_yb, neg_instance_yb, neg_id = neg_data

    # Combine positive and negative data
    combined_xb = pos_xb + neg_xb
    combined_yb = torch.cat([pos_yb, neg_yb])
    combined_instance_yb = pos_instance_yb + neg_instance_yb
    combined_id = pos_id + neg_id

    return combined_xb, combined_yb, combined_instance_yb, combined_id



def default_train():
    global val_loss_best
    
    # Training phase
    bagmodel.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for (data, yb, instance_yb, id) in tqdm(train_dl, total=len(train_dl)): 
        xb, yb = data, yb.cuda()
        
        optimizer.zero_grad()
        
        outputs, _, _, _, _ = bagmodel(xb)

        loss = loss_func(outputs, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)
        predicted = (outputs > .5).float()
        total += yb.size(0)
        if len(label_columns) == 1:  # Binary or single-label classification
            correct += (predicted == yb).sum().item()
        else:  # Multi-label classification
            correct += ((predicted == yb).sum(dim=1) == len(label_columns)).sum().item()
            
    train_loss = total_loss / total
    train_acc = correct / total

    # Evaluation phase
    bagmodel.eval()
    total_val_loss = 0.0
    total = 0
    correct = 0
    all_targs = []
    all_preds = []
    with torch.no_grad():
        for (data, yb, instance_yb, id) in tqdm(val_dl, total=len(val_dl)): 
            xb, yb = data, yb.cuda()

            outputs, _, _, _, _ = bagmodel(xb)
            loss = loss_func(outputs, yb)
            
            total_val_loss += loss.item() * len(xb)
            predicted = (outputs > .5).float()
            total += yb.size(0)
            if len(label_columns) == 1:  # Binary or single-label classification
                correct += (predicted == yb).sum().item()
            else:  # Multi-label classification
                correct += ((predicted == yb).sum(dim=1) == len(label_columns)).sum().item()
            
            # Confusion Matrix data
            all_targs.extend(yb.cpu().numpy())
            if len(predicted.size()) == 0:
                predicted = predicted.view(1)
            all_preds.extend(predicted.cpu().detach().numpy())

    val_loss = total_val_loss / total
    val_acc = correct / total
    
    train_losses_over_epochs.append(train_loss)
    valid_losses_over_epochs.append(val_loss)
    
    print(f"Epoch {epoch+1} | Acc   | Loss")
    print(f"Train   | {train_acc:.4f} | {train_loss:.4f}")
    print(f"Val     | {val_acc:.4f} | {val_loss:.4f}")
        
    # Save the model
    if val_loss < val_loss_best:
        val_loss_best = val_loss  # Update the best validation accuracy
        save_state(epoch, label_columns, train_acc, val_loss, val_acc, model_folder, model_name, bagmodel, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs)
        print("Saved checkpoint due to improved val_loss")


if __name__ == '__main__':

    # Config
    model_name = 'ITS2CLR-test'
    encoder_arch = 'resnet18'
    dataset_name = 'export_12_26_2023'
    label_columns = ['Has_Malignant']
    instance_columns = [] #['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 350
    batch_size = 5
    min_bag_size = 2
    max_bag_size = 20
    lr = 0.001
    
    #ITS2CLR Config
    feature_extractor_train_count = 10
    initial_ratio = 0.0  #100% negitive bags
    final_ratio = 0.8  #20% negitive bags
    total_epochs = 200
    warmup_epochs = 5

    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    #export_location = '/home/paperspace/cadbusi-LFS/export_09_28_2023/'
    #cropped_images = f"/home/paperspace/Temp_Data/{img_size}_images/"
    

    # Get Training Data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_labels = len(label_columns)

    print("Training Data...")
    # Create datasets
    positive_train_dataset = TUD.Subset(ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='positive'),list(range(0,100)))
    negative_train_dataset = TUD.Subset(ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='negative'),list(range(0,100)))
    train_dataset = TUD.Subset(ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='all'),list(range(0,100)))
    dataset_val = TUD.Subset(ITS2CLR_Dataset(bags_val, save_processed=False),list(range(0,100)))
    
    #positive_train_dataset = ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='positive')
    #negative_train_dataset = ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='negative')
    #train_dataset = ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='all')
    #dataset_val = ITS2CLR_Dataset(bags_val, train=False)

            
    # Create data loaders
    pos_train_dl =  TUD.DataLoader(positive_train_dataset, batch_size=batch_size, collate_fn = collate_bag, drop_last=True, shuffle = True)
    neg_train_dl =  TUD.DataLoader(negative_train_dataset, batch_size=batch_size, collate_fn = collate_bag, drop_last=True, shuffle = True)
    train_dl =  TUD.DataLoader(train_dataset, batch_size=batch_size, collate_fn = collate_bag, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_bag, drop_last=True)

    
    encoder = create_timm_body(encoder_arch)
    nf = num_features_model(nn.Sequential(*encoder.children()))
    
    # bag aggregator
    aggregator = ABMIL_aggregate( nf = nf, num_classes = num_labels, pool_patches = 6, L = 128)

    # total model
    bagmodel = EmbeddingBagModel(encoder, aggregator, num_classes = num_labels).cuda()
    total_params = sum(p.numel() for p in bagmodel.parameters())
    print(f"Total Parameters: {total_params}")
    
    # Define the adaptive average pooling layer
    adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(device)
        
        
    optimizer = Adam(bagmodel.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    supcon_loss = SupConLoss(temperature=0.07, pair_mode = 2, contrast_mode='all', base_temperature=0.07)
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    epoch_start = 0
    

    
    # Check if the model already exists
    model_folder = f"{env}/models/{model_name}/"
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"
    
    if os.path.exists(model_path):
        bagmodel.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"Loaded pre-existing model from {model_name}")
        
        with open(stats_path, 'rb') as f:
            saved_stats = pickle.load(f)
            train_losses_over_epochs = saved_stats['train_losses']
            valid_losses_over_epochs = saved_stats['valid_losses']
            epoch_start = saved_stats['epoch']
            val_loss_best = saved_stats['val_loss']
    else:
        print(f"{model_name} does not exist, creating new instance")
        os.makedirs(model_folder, exist_ok=True)
        val_loss_best = 99999


    # Training loop
    for epoch in range(epoch_start, total_epochs):
        if epoch % (1 + feature_extractor_train_count) == 0:
            
            print('Training Default')
            default_train()
        else:
            print('Training Feature Extractor')
            # Generalized Supervised Contrastive Learning phase
            # Freeze the aggregator and unfreeze the encoder
            for param in aggregator.parameters():
                param.requires_grad = False
            for param in encoder.parameters():
                param.requires_grad = True
            

            # Get difficualy ratio
            current_ratio = spl_scheduler(epoch, total_epochs, initial_ratio, final_ratio)
            print(f'Current Bag Ratio: {current_ratio:.2f}')
            pos_batch_size = int(current_ratio * batch_size)
            neg_batch_size = batch_size - pos_batch_size

            # Iterator for positive and negative data loaders
            pos_data_iter = iter(pos_train_dl)
            neg_data_iter = iter(neg_train_dl)

            # Determine the total number of batches
            total_batches = max(len(pos_train_dl), len(neg_train_dl))
            
            epoch_loss = 0.0

            # Iterate over the training data
            for _ in tqdm(range(total_batches)):
                # Fetch batches from positive and negative data loaders
                pos_data = next(pos_data_iter, None)
                neg_data = next(neg_data_iter, None)

                if pos_data is None or neg_data is None:
                    break  # Break if either data loader is exhausted

                # Combine positive and negative batches
                combined_data = combine_pos_neg_data(pos_data, neg_data, pos_batch_size, neg_batch_size)
                xb, yb, instance_yb, id = combined_data
            
                yb = yb.cuda().view(-1)
                optimizer.zero_grad()

                # Generate pseudo labels from attention scores
                logits, sm, yhat_ins, att_sc, encoder_outputs = bagmodel(xb)
                
                pseudo_labels = generate_pseudo_labels(att_sc)
                
                # Apply adaptive average pooling to the feature maps
                encoder_outputs = adaptive_avg_pool(encoder_outputs)

                # Calculate the correct number of features per bag
                split_sizes = [bag.size(0) for bag in xb]
                features_per_bag = torch.split(encoder_outputs, split_sizes, dim=0)

                # Initialize a list to hold losses for each bag
                losses = []

                # Iterate over each bag's features and corresponding pseudo labels
                for bag_features, bag_pseudo_labels_list in zip(features_per_bag, pseudo_labels):
                    # Ensure bag_features and bag_pseudo_labels are on the same device
                    bag_features = bag_features.to(device)
                    bag_pseudo_labels_tensor = bag_pseudo_labels_list.to(device)

                    bag_features = F.normalize(bag_features, dim=1)
                    
                    #print("Shape of features:", bag_features.shape)
                    #print("Shape of teacher_probs (pseudo labels):", bag_pseudo_labels_tensor.shape)

                    # Compute the loss for the current bag
                    bag_loss = supcon_loss(bag_features, bag_pseudo_labels_tensor)
                    
                    # Store the loss
                    losses.append(bag_loss)

                # Combine losses for all bags
                total_loss = torch.mean(torch.stack(losses))
                epoch_loss += total_loss.item()
                #print("Batch Loss: ", total_loss)

                # Backward pass
                total_loss.backward()
                optimizer.step()
                
            # Calculate the average loss for the epoch
            epoch_loss /= total_batches
            print(f'Loss: {epoch_loss:3f}')

            # After the contrastive update, unfreeze the aggregator and encoder
            for param in aggregator.parameters():
                param.requires_grad = True
            for param in encoder.parameters():
                param.requires_grad = True
                
                
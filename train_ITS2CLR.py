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
    
    
def collate_custom(batch):
    batch_data = []
    batch_labels = []
    batch_ids = []  # List to store bag IDs

    for sample in batch:
        image_data, labels, bag_id = sample
        batch_data.append(image_data)
        batch_labels.append(labels)  # labels are already tensors
        batch_ids.append(bag_id)  # Append the bag ID

    # Using torch.stack for labels to handle multiple labels per bag
    out_labels = torch.stack(batch_labels).cuda()

    # Converting to a tensor
    out_ids = torch.tensor(batch_ids, dtype=torch.long).cuda()

    return batch_data, out_labels, out_ids

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
        
        return logits, saliency_maps, yhat_instances, attention_scores


def generate_pseudo_labels(inputs, threshold = .5):
    pseudo_labels = []
    for tensor in inputs:
        # Calculate the dynamic threshold for each tensor
        """pseudo_labels_tensor = (tensor >= threshold).float()
        pseudo_labels_list = pseudo_labels_tensor.tolist()
        pseudo_labels.append(pseudo_labels_list)"""
        
        # Calculate the dynamic threshold for each tensor
        threshold = 1 / tensor.size(0)
        pseudo_labels_tensor = (tensor > threshold).float()
        pseudo_labels.append(pseudo_labels_tensor)
    
    print("attention scores: ", inputs)    
    print("pseudo_labels: ", pseudo_labels)
    return pseudo_labels

# use raw output? still convert to 0 or 1 (Pseudo labels)
# Need instance labels 
# In the begginnning only do negitivebags in the loss
# Threshold .1? .1 or .9 prediction would pass. Slowly decreaing that threshold 
# Two seperate throholds for 1 and 0?
# DO NOT USE POSITIVE CASES IN BEGINNING
# Rank all images in batch, take best ten percent in the beginning 
# Paramter that controls the amount of positive cases that show up in the numerator in the loss function (IDEALLY)
# Just dont compute it initially?
# You should use Yhats instead, they are all from 0 to 1 individually

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
        print("mask", mask)
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        if self.mask_uncertain_neg:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+ 1e-10)
        else:
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) )

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        
        
        print("anchor_feature", anchor_feature)
        print("anchor_dot_contrast", anchor_dot_contrast)
        print("logits", logits)
        print("torch.exp(logits)", torch.exp(logits))
        print("mean_log_prob_pos", mean_log_prob_pos)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if torch.isinf(mask).any():
            print("There are NaNs in the mask tensor.")
        else:
            print("No NaNs in the mask tensor.")
            
        if torch.isinf(log_prob).any():
            print("There are NaNs in the log_prob tensor.")
        else:
            print("No NaNs in the log_prob tensor.")
            
        if torch.isnan(mean_log_prob_pos).any():
            print("There are NaNs in the mean_log_prob_pos tensor.")
        else:
            print("No NaNs in the mean_log_prob_pos tensor.")
            
        print("LOSS", loss)
        print("anchor_count", anchor_count)
        print("batch_size", batch_size)
        loss = loss.view(anchor_count, batch_size).mean()

        print("LOSS", loss)

        return loss



# Training vars
val_acc_best = -1 

def default_train():
    global val_acc_best
    
    # Training phase
    bagmodel.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for (data, yb, _) in tqdm(train_dl, total=len(train_dl)): 
        xb, yb = data, yb.cuda()
        
        optimizer.zero_grad()
        
        outputs, _, _, _ = bagmodel(xb)

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
        for (data, yb, _) in tqdm(val_dl, total=len(val_dl)): 
            xb, yb = data, yb.cuda()

            outputs, _, _, _ = bagmodel(xb)
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
    if val_acc > val_acc_best:
        val_acc_best = val_acc  # Update the best validation accuracy
        save_state(epoch, train_acc, val_acc, model_folder, model_name, bagmodel, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs)
        print("Saved checkpoint due to improved val_acc")


if __name__ == '__main__':

    # Config
    # Config
    model_name = 'test'
    encoder_arch = 'resnet18'
    dataset_name = 'export_11_11_2023'
    label_columns = ['Has_Malignant']
    img_size = 350
    batch_size = 5
    min_bag_size = 2
    max_bag_size = 20
    epochs = 500
    lr = 0.001

    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    #export_location = '/home/paperspace/cadbusi-LFS/export_09_28_2023/'
    #cropped_images = f"/home/paperspace/Temp_Data/{img_size}_images/"
    

    # Get Training Data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, cropped_images, img_size, min_bag_size, max_bag_size)


    print("Training Data...")
    # Create datasets
    dataset_train = TUD.Subset(BagOfImagesDataset(bags_train, save_processed=False),list(range(0,100)))
    dataset_val = TUD.Subset(BagOfImagesDataset(bags_val, save_processed=False),list(range(0,100)))
    #dataset_train = BagOfImagesDataset(bags_train, save_processed=False)
    #dataset_val = BagOfImagesDataset(bags_val, train=False)

            
    # Create data loaders
    train_dl =  TUD.DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)



    # Check if the model already exists
    model_folder = f"{env}/models/{model_name}/"
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"

    
    encoder = create_timm_body(encoder_arch)

    nf = num_features_model(nn.Sequential(*encoder.children()))
    # bag aggregator
    aggregator = ABMIL_aggregate( nf = nf, num_classes = 1, pool_patches = 3, L = 128)

    # total model
    bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
    total_params = sum(p.numel() for p in bagmodel.parameters())
    print(f"Total Parameters: {total_params}")
        
        
    optimizer = Adam(bagmodel.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    supcon_loss = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    epoch_start = 0
    

    
    if os.path.exists(model_path):
        bagmodel.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"Loaded pre-existing model from {model_name}")
        
        with open(stats_path, 'rb') as f:
            saved_stats = pickle.load(f)
            train_losses_over_epochs = saved_stats['train_losses']
            valid_losses_over_epochs = saved_stats['valid_losses']
            epoch_start = saved_stats['epoch']
            val_acc_best = saved_stats.get('val_acc', -1)  # If 'val_acc' does not exist, default to -1
    else:
        print(f"{model_name} does not exist, creating new instance")
        os.makedirs(model_folder, exist_ok=True)
        val_acc_best = -1 


    # Training loop
    for epoch in range(epoch_start, epochs):
        if epoch % (1 + feature_extractor_train_count) == 0:
            
            print('Training Default')
            default_train()
        else:
            print('Training Feature Extractor')
            # Supervised Contrastive Learning phase
            # Freeze the aggregator and unfreeze the encoder
            for param in aggregator.parameters():
                param.requires_grad = False
            for param in encoder.parameters():
                param.requires_grad = True

            # Iterate over the training data
            for (data, _, _) in tqdm(train_dl, total=len(train_dl)):
                xb = data
                optimizer.zero_grad()

                # Generate pseudo labels from attention scores
                logits, sm, yhat_ins, att_sc = bagmodel(xb)
                
                pseudo_labels = generate_pseudo_labels(att_sc)

                # Forward pass through the encoder only
                outputs = encoder(torch.cat(xb, dim=0))

                # Calculate the correct number of features per bag
                split_sizes = [bag.size(0) for bag in xb]
                features_per_bag = torch.split(outputs, split_sizes, dim=0)

                # Initialize a list to hold losses for each bag
                losses = []

                # Iterate over each bag's features and corresponding pseudo labels
                for bag_features, bag_pseudo_labels_list in zip(features_per_bag, pseudo_labels):
                    # Ensure bag_features and bag_pseudo_labels are on the same device
                    bag_features = bag_features.to(device)
                    bag_pseudo_labels_tensor = bag_pseudo_labels_list.to(device)
                    #bag_pseudo_labels_tensor = torch.tensor(bag_pseudo_labels_list, dtype=torch.float32).to(device)
                    
                    zjs = F.normalize(bag_features, dim=1)

                    bag_features = torch.cat([zjs.unsqueeze(1), zjs.unsqueeze(1)], dim=1)
                    
                    print("final: ", bag_pseudo_labels_tensor)
                    print("bag_features: ", bag_features.shape)
                    
                    # Compute the loss for the current bag
                    bag_loss = supcon_loss(bag_features, bag_pseudo_labels_tensor)
                    
                    # Store the loss
                    losses.append(bag_loss)

                # Combine losses for all bags
                total_loss = torch.mean(torch.stack(losses))
                print(total_loss)

                # Backward pass
                total_loss.backward()
                optimizer.step()

            # After the contrastive update, unfreeze the aggregator and encoder
            for param in aggregator.parameters():
                param.requires_grad = True
            for param in encoder.parameters():
                param.requires_grad = True
                
                
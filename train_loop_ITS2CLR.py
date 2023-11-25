import os, pickle
from timm import create_model
from fastai.vision.all import *
import torch.utils.data as TUD
from fastai.vision.learner import _update_first_layer
from tqdm import tqdm
from torch import nn
from training_eval import *
from torch.optim import Adam
from data_prep import *
from model_ABMIL import *
from model_TransMIL import *
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# this function is used to cut off the head of a pretrained timm model and return the body
def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or function")


def collate_custom(batch):
    batch_data = []
    batch_labels = []
    batch_ids = []  # List to store bag IDs

    for sample in batch:
        image_data, label, bag_id = sample
        batch_data.append(image_data)
        batch_labels.append(label)
        batch_ids.append(bag_id)  # Append the bag ID

    out_labels = torch.tensor(batch_labels).cuda()
    out_ids = torch.tensor(batch_ids).cuda()  # Convert bag IDs to a tensor
    
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
        
        return logits.squeeze(1), saliency_maps, yhat_instances, attention_scores


def generate_pseudo_labels(attention_scores):
    pseudo_labels = []
    for tensor in attention_scores:
        # Calculate the dynamic threshold for each tensor
        threshold = 1 / tensor.size(0)
        # Apply the threshold to each tensor
        pseudo_labels_tensor = (tensor > threshold).float()
        pseudo_labels.append(pseudo_labels_tensor)
    return pseudo_labels



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
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
                # mask = torch.eq(labels, labels.T).float().to(device)
                pass

            # only take the positive into account
            elif self.pair_mode==1:
                # mask = (torch.eq(labels, labels.T) * (labels == 1)).float().to(device)
                mask[labels[:, 0] == 0, :] = 0
                mask[:, labels[:, 0] == 0] = 0

            # only take the negative into account
            elif self.pair_mode==2:
                # mask = (torch.eq(labels, labels.T) * (labels == 0)).float().to(device)
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
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast
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
        predicted = torch.round(outputs).squeeze()
        total += yb.size(0)
        correct += predicted.eq(yb.squeeze()).sum().item()

    train_loss = total_loss / total
    train_acc = correct / total
    
    val_check(train_acc, train_loss)



def val_check(train_acc, train_loss):
    global val_acc_best
    
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
            predicted = torch.round(outputs).squeeze() 
            total += yb.size(0)
            correct += predicted.eq(yb.squeeze()).sum().item()
            
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
    model_name = 'ITS2CLR_2'
    img_size = 350
    batch_size = 5
    feature_extractor_train_count = 5
    min_bag_size = 2
    max_bag_size = 18
    epochs = 500
    lr = 0.001

    # Paths
    export_location = 'D:/DATA/CASBUSI/exports/export_11_11_2023/'
    cropped_images = f"F:/Temp_SSD_Data/{img_size}_images/"
    #export_location = '/home/paperspace/cadbusi-LFS/export_09_28_2023/'
    #cropped_images = f"/home/paperspace/Temp_Data/{img_size}_images/"
    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
    image_data = pd.read_csv(f'{export_location}/ImageData.csv')
    

    
    bags_train, bags_val = prepare_all_data(export_location, case_study_data, breast_data, image_data, cropped_images, img_size, min_bag_size, max_bag_size)



    print("Training Data...")
    # Create datasets
    #dataset_train = TUD.Subset(BagOfImagesDataset(bags_train),list(range(0,100)))
    #dataset_val = TUD.Subset(BagOfImagesDataset(bags_val),list(range(0,100)))
    dataset_train = BagOfImagesDataset(bags_train, save_processed=False)
    dataset_val = BagOfImagesDataset(bags_val, train=False)

            
    # Create data loaders
    train_dl =  TUD.DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)


    encoder = create_timm_body('resnet18')
    nf = num_features_model( nn.Sequential(*encoder.children()))
    
    # bag aggregator
    aggregator = ABMIL_aggregate( nf = nf, num_classes = 1, pool_patches = 3, L = 128)
    #aggregator = TransMIL(dim_in=nf, dim_hid=512, n_classes=1)

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
            val_acc_best = saved_stats.get('val_acc', -1)  # If 'val_acc' does not exist, default to -1
    else:
        print(f"{model_name} does not exist, creating new instance")
        os.makedirs(model_folder, exist_ok=True)
        val_acc_best = -1 


    # Training loop
    for epoch in range(epoch_start, epochs):
        if epoch % (1 + feature_extractor_train_count) == 0:
            val_check(0, 0)
            
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
                for bag_features, bag_pseudo_labels in zip(features_per_bag, pseudo_labels):
                    # Ensure bag_features and bag_pseudo_labels are on the same device
                    bag_features = bag_features.to(device)
                    bag_pseudo_labels = bag_pseudo_labels.to(device)

                    # Compute the loss for the current bag
                    bag_loss = supcon_loss(bag_features, bag_pseudo_labels)

                    # Store the loss
                    losses.append(bag_loss)

                # Combine losses for all bags
                total_loss = torch.mean(torch.stack(losses))

                # Backward pass
                total_loss.backward()
                optimizer.step()

            # After the contrastive update, unfreeze the aggregator and encoder
            for param in aggregator.parameters():
                param.requires_grad = True
            for param in encoder.parameters():
                param.requires_grad = True
                
                
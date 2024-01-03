import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from archs.save_arch import *
from torch.optim import Adam
from torch.cuda.amp import autocast
from data.format_data import *
from archs.model_FC import *
from archs.backbone import create_timm_body
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
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
        super(EmbeddingBagModel,self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.num_classes = num_classes

    def forward(self, input):
        num_bags = len(input)  # input = [bag #, image #, channel, height, width]
        
        # Concatenate all bags into a single tensor for batch processing
        all_images = torch.cat(input, dim=0)  # Shape: [Total images in all bags, channel, height, width]
        
        # Calculate the embeddings for all images in one go
        h_all = self.encoder(all_images)
        
        # Split the embeddings back into per-bag embeddings
        split_sizes = [bag.size(0) for bag in input]
        h_per_bag = torch.split(h_all, split_sizes, dim=0)
        logits = torch.empty(num_bags, self.num_classes).cuda()
        yhat_instances, attention_scores = [], []
        
        for i, h in enumerate(h_per_bag):
            # Receive values from the aggregator
            yhat_bag, yhat_ins, att_sc = self.aggregator(h)
            
            logits[i] = yhat_bag
            yhat_instances.append(yhat_ins)
            attention_scores.append(att_sc)
        
        return logits, yhat_instances, attention_scores



class GenSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(GenSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        '''
        Args:
            feats: (anchor_features, contrast_features), each: [N, feat_dim]
            labels: (anchor_labels, contrast_labels) each: [N, num_cls]
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
        loss = loss.mean()

        return loss
    

def mix_fn(x, y, alpha, kind):
    if kind == 'mixup':
        return mixup_data_per_bag(x, y, alpha)
    elif kind == 'cutmix':
        return cutmix_data(x, y, alpha)
    elif kind == 'mixup_cutmix':
        if np.random.rand(1)[0] > 0.5:
            return mixup_data_per_bag(x, y, alpha)
        else:
            return cutmix_data(x, y, alpha)
    else:
        raise ValueError()


def mix_target(y_a, y_b, lam, num_classes):
    # Debug: Print shapes and types of y_a, y_b, and lam
    #print("y_a original:", y_a, "Shape:", y_a.shape, "Type:", y_a.dtype)
    #print("y_b original:", y_b, "Shape:", y_b.shape, "Type:", y_b.dtype)
    #print("lam:", lam, "Type:", lam.dtype)

    # Ensure y_a and y_b are tensors with at least 1 dimension
    y_a = y_a if y_a.dim() > 0 else y_a.unsqueeze(0)
    y_b = y_b if y_b.dim() > 0 else y_b.unsqueeze(0)

    # Debug: Print shapes after potential unsqueeze operation
    #print("y_a adjusted:", y_a.shape)
    #print("y_b adjusted:", y_b.shape)

    # Mix the labels based on lambda
    mixed_labels = lam * y_a + (1 - lam) * y_b

    # Debug: Print the mixed labels
    #print("Mixed labels:", mixed_labels, "Shape:", mixed_labels.shape)

    return mixed_labels


def mixup_data_per_bag(x, y, alpha=1.0):
    '''Apply mixup to each bag separately'''
    mixed_x_bags = []
    y_a_list = []
    y_b_list = []
    lam_list = []

    for bag_images, bag_labels in zip(x, y):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        bag_size = bag_images.size(0)
        index = torch.randperm(bag_size, device=bag_images.device)

        mixed_bag_images = lam * bag_images + (1 - lam) * bag_images[index, :]
        mixed_x_bags.append(mixed_bag_images)

        # Since bag_labels is a scalar, use it directly for y_a and y_b
        y_a, y_b = bag_labels, bag_labels

        y_a_list.append(y_a.unsqueeze(0))  # Add a dimension
        y_b_list.append(y_b.unsqueeze(0))  # Add a dimension
        lam_list.append(torch.full((1,), lam, device=bag_images.device))

    # Concatenate all bags back into a batch
    mixed_x = torch.cat(mixed_x_bags, dim=0)
    y_a_all = torch.cat(y_a_list, dim=0)
    y_b_all = torch.cat(y_b_list, dim=0)
    lam_all = torch.cat(lam_list, dim=0)

    return mixed_x, y_a_all, y_b_all, lam_all


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    bsz = x.size()[0]
    index = torch.randperm(bsz, device=x.device)
    
    bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
    mixed_x = x.detach().clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = mixed_x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if __name__ == '__main__':

    # Config
    teacher_name = 'FC_test_2'
    model_name = 'test'
    encoder_arch = 'resnet18'
    dataset_name = 'export_11_11_2023'
    label_columns = ['Has_Malignant', 'Has_Benign']
    instance_columns = ['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 350
    batch_size = 5
    min_bag_size = 2
    max_bag_size = 20
    tempurature = 0.07
    epochs = 500
    class_num = 1
    KD_alpha = .5 # ratio of training from the teacher model
    KD_temp = 1 # Smoothing (Tempurature) 
    lr = 0.001

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
    dataset_train = TUD.Subset(ContrastiveLoader(bags_train, save_processed=False),list(range(0,100)))
    dataset_val = TUD.Subset(ContrastiveLoader(bags_val, save_processed=False),list(range(0,100)))
    #dataset_train = ContrastiveLoader(bags_train, save_processed=False)
    #dataset_val = ContrastiveLoader(bags_val, train=False)
            
    # Create data loaders
    train_dl =  TUD.DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)

    
    # Check if the model already exists
    model_folder = f"{env}/models/{model_name}/"
    model_path = f"{model_folder}/{model_name}.pth"
    teacher_path = f"{model_folder}/{teacher_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"

    encoder = create_timm_body(encoder_arch)
    nf = num_features_model( nn.Sequential(*encoder.children()))
    criterion = GenSupConLoss(temperature=tempurature)
    
    # bag aggregator
    aggregator = FC_aggregate( nf = nf, num_classes = num_labels, L = 128, fc_layers=[256, 64], dropout = .5)

    # total model
    bagmodel = EmbeddingBagModel(encoder, aggregator, num_classes = num_labels).cuda()
    teacher_model = EmbeddingBagModel(encoder, aggregator).cuda()
    total_params = sum(p.numel() for p in bagmodel.parameters())
    print(f"Total Parameters: {total_params}")
        
        
    optimizer = Adam(bagmodel.parameters(), lr=lr)
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    epoch_start = 0
    
    if os.path.exists(teacher_path):
        teacher_model.load_state_dict(torch.load(teacher_path))
        teacher_model.eval()
        print(f"Loaded pre-existing teacher model from {model_name}")
    
    
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
        bagmodel.train()
        total_loss = 0.0
        for (data, yb, _) in tqdm(train_dl, total=len(train_dl)):
            xb, yb = data, yb.cuda()
            bsz = yb.shape[0]
            im_q, im_k = xb
            
            im_q = im_q.cuda(non_blocking=True)
            im_k = im_k.cuda(non_blocking=True)
            yb = yb.cuda(non_blocking=True)

            # Apply Mixup/Cutmix Augmentation
            im_q, y0a, y0b, lam0 = mix_fn(im_q, yb, alpha=1.0, kind='mixup_cutmix')
            im_k, y1a, y1b, lam1 = mix_fn(im_k, yb, alpha=1.0, kind='mixup_cutmix')
            images = torch.cat([im_q, im_k], dim=0)
            l_q = mix_target(y0a, y0b, lam0, class_num)
            l_k = mix_target(y1a, y1b, lam1, class_num)
            
            # Get teacher results
            with torch.no_grad():
                with autocast():
                    teacher_logits, _, _ = teacher_model(xb)
                    preds = F.softmax(teacher_logits / KD_temp, dim=1)
                    teacher_q, teacher_k = torch.split(preds, [bsz, bsz], dim=0)

            

            # Extract features using your model
            features, _, _ = bagmodel(images)
            features = torch.split(features, [bsz, bsz], dim=0)
            
            optimizer.zero_grad()
            
            if KD_alpha == float('inf'): # only learn from teacher's prediction
                loss = criterion(features, [teacher_q, teacher_k])
            else:
                loss = criterion(features, [l_q, l_k]) + KD_alpha * criterion(features, [teacher_q, teacher_k])
                
            print(loss)
            
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
        total_val_acc = 0.0
        total = 0
        correct = 0
        all_targs = []
        all_preds = []
        with torch.no_grad():
            for (data, yb, _) in tqdm(val_dl, total=len(val_dl)): 
                xb, yb = data, yb.cuda()

                outputs, _, _ = bagmodel(xb)
                loss = nn.BCELoss(outputs, yb)
                
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
import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch.cuda.amp import autocast
import timm
from torch import nn
from archs.model_ITS2CLR import *
from data.ITS2CLR_util import *
import wandb
from archs.save_arch import *
from torch.optim import Adam
from data.format_data import *

import sys
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
try:
    import wandb
except ImportError:
    pass


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
        
         # Process images for 'query' and 'key'
        image_data_q = torch.stack([self.tsfms(Image.open(fn).convert("RGB")) for fn in files_this_bag])
        image_data_k = torch.stack([self.tsfms(Image.open(fn).convert("RGB")) for fn in files_this_bag])

           
        # Save processed images if required
        if self.save_processed:
            
            # Process images
            image_data = torch.stack([self.tsfms(Image.open(fn).convert("RGB")) for fn in files_this_bag])
            image_data = image_data.cuda()  # Move to GPU if CUDA is available
        
            save_folder = os.path.join(env, 'processed_images')  
            os.makedirs(save_folder, exist_ok=True)
            for idx, img_tensor in enumerate(image_data):
                img_save_path = os.path.join(save_folder, f'bag_{actual_id}_img_{idx}.png')
                img_tensor = unnormalize(img_tensor)
                img = TF.to_pil_image(img_tensor.cpu().detach())
                img.save(img_save_path)

        # Create a tensor for bag-level labels for each image
        num_images = len(files_this_bag)
        bag_labels_per_image = torch.full((num_images,), bag_labels[0], dtype=torch.float32)

        # Convert instance labels to a tensor, using -1 for None
        instance_labels_tensors = [torch.tensor(labels, dtype=torch.float32) if labels != [None] else torch.tensor([-1], dtype=torch.float32) for labels in instance_labels]

        return (image_data_q, image_data_k), bag_labels_per_image, instance_labels_tensors, actual_id

    
    def __len__(self):
        return len(self.unique_bag_ids)
    
    def n_features(self):
        return self.data.size(1)
    
    
def collate_custom(batch):
    batch_data = []
    batch_bag_labels = []
    batch_instance_labels = []
    batch_ids = []  # List to store bag IDs

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



def train(loader, model, teacher, criterion, optimizer, epoch, args):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    
    for idx, (images, targets, instance_yb, id) in enumerate(loader):
        data_time.update(time.time() - end)
        warmup_learning_rate(args, epoch, idx, len(loader), optimizer)
        
        bsz = targets.shape[0]
        im_q, im_k = images[0]
        if torch.cuda.is_available():
            im_q = im_q.cuda(non_blocking=True)
            im_k = im_k.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
        if args.mix: # image-based regularizations
            im_q, y0a, y0b, lam0 = mix_fn(im_q, targets, args.mix_alpha, args.mix)
            im_k, y1a, y1b, lam1 = mix_fn(im_k, targets, args.mix_alpha, args.mix)
            images = torch.cat([im_q, im_k], dim=0)
            l_q = mix_target(y0a, y0b, lam0, num_labels)
            l_k = mix_target(y1a, y1b, lam1, num_labels)
        else:
            images = torch.cat([im_q, im_k], dim=0)
            l_q = F.one_hot(targets, num_labels)
            l_k = l_q
        
        if teacher: # KD
            with torch.no_grad():
                with autocast():
                    preds = F.softmax(teacher(images) / args.KD_temp, dim=1)
                    teacher_q, teacher_k = torch.split(preds, [bsz, bsz], dim=0)
                
        # forward
        features = model(images)
        features = torch.split(features, [bsz, bsz], dim=0)

        if teacher:
            if args.KD_alpha == float('inf'): # only learn from teacher's prediction
                loss = criterion(features, [teacher_q, teacher_k])
            else:
                loss = criterion(features, [l_q, l_k]) + args.KD_alpha * criterion(features, [teacher_q, teacher_k])
        else: # no KD
            loss = criterion(features, [l_q, l_k])

        
        losses.update(loss.item(), bsz)
        # backwaqrd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
            
    res = {
        'trn_loss': losses.avg,
        'learning_rate': get_learning_rate(optimizer)
    }
    return res



class Args:
    def __init__(self, warm, warm_epochs, learning_rate, lr_decay_rate, epochs, warmup_from, cosine, lr_decay_epochs, mix, mix_alpha, KD_temp, KD_alpha, print_freq, teacher_path, teacher_ckpt):
        self.warm = warm
        self.warm_epochs = warm_epochs
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.epochs = epochs
        self.warmup_from = warmup_from
        self.cosine = cosine
        self.lr_decay_epochs = lr_decay_epochs
        self.mix = mix
        self.mix_alpha = mix_alpha
        self.KD_temp = KD_temp
        self.KD_alpha = KD_alpha
        self.print_freq = print_freq
        self.teacher_path = teacher_path
        self.teacher_ckpt = teacher_ckpt

if __name__ == '__main__':

    # Config
    model_name = 'GenSCL-test'
    encoder_arch = 'resnet18'
    dataset_name = 'export_12_26_2023'
    label_columns = ['Has_Malignant']
    instance_columns = [] #['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 350
    batch_size = 5
    min_bag_size = 2
    max_bag_size = 20

    model_folder = f"{env}/models/{model_name}/"
    
    args = Args(
        warm=True,
        warm_epochs=5,
        learning_rate=0.01,
        lr_decay_rate=0.1,
        epochs=50,
        warmup_from=0.001,
        cosine=True,
        lr_decay_epochs=[30, 40],
        mix='mixup',
        mix_alpha=0.2,
        KD_temp=4,
        KD_alpha=0.9,
        print_freq=10,
        teacher_path=model_folder,
        teacher_ckpt='teacher_ITS2CLR.pth'
    )
    
    
    #ITS2CLR Config
    feature_extractor_train_count = 10
    initial_ratio = 0.0  #100% negitive bags
    final_ratio = 0.8  #20% negitive bags
    total_epochs = 200
    warmup_epochs = 25

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
    #positive_train_dataset = TUD.Subset(ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='positive'),list(range(0,100)))
    #negative_train_dataset = TUD.Subset(ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='negative'),list(range(0,100)))
    #train_dataset = TUD.Subset(ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='all'),list(range(0,100)))
    #dataset_val = TUD.Subset(BagOfImagesDataset(bags_val, save_processed=False),list(range(0,100)))
    
    train_dataset = ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='all')
    dataset_val = BagOfImagesDataset(bags_val, train=False)

            
    # Create data loaders
    train_dl =  TUD.DataLoader(train_dataset, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)

    
    # Create model
    model = SupConResNet(name='resnet18')
    criterion = GenSupConLoss(temperature=0.07)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    # Load teacher model
    """teacher = timm.create_model('resnet18', pretrained=False)
    teacher.reset_classifier(num_labels)

    # Assuming the first convolutional layer is named 'conv1'
    out_ch = teacher.conv1.out_channels
    teacher.conv1 = torch.nn.Conv2d(3, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    teacher_ckpt = torch.load(Path(args.teacher_path) / args.teacher_ckpt, map_location='cpu')
    teacher.load_state_dict(teacher_ckpt['state_dict'])
    teacher.eval()"""
    teacher = None
    
    # Resume
    """load_fn = Path(args.save_root)/args.desc/f'ckpt_{args.resume}.pth'
    ckpt = torch.load(load_fn, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    print(f'=> Successfully loading {load_fn}!')
    args.start_epoch = ckpt['epoch'] + 1"""
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
        
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    epoch_start = 0
    

    
    # Check if the model already exists
    """model_folder = f"{env}/models/{model_name}/"
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
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
        val_loss_best = 99999"""


    # Training loop
    for epoch in range(epoch_start, total_epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        res = train(train_dl, model, teacher, criterion, optimizer, epoch, args)
        time2 = time.time()
        print(f'epoch {epoch}, total time {format_time(time2 - time1)}')
        
        wandb.log(res, step=epoch)
        
        """if (epoch % args.save_freq == 0) and not args.debug:
            save_fn = save_dir/f'ckpt_{epoch}.pth'
            save_model(model, optimizer, args, epoch, save_fn)"""
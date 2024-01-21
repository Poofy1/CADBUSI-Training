import os
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch import nn
from archs.model_GenSCL import *
from data.ITS2CLR_util import *
import wandb
from archs.save_arch import *
from torch.optim import Adam
from data.format_data import *
import sys
import time


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
        
        self.images = []  # List to store individual images
        self.bag_labels = []  # List to store corresponding bag labels for each image

        # Iterate over each bag and add images and labels to the lists
        for bag_id in self.unique_bag_ids:
            bag_info = bags_dict[bag_id]
            for img_path in bag_info['images']:
                self.images.append(img_path)
                self.bag_labels.append(bag_info['bag_labels'][0])

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
        img_path = self.images[index]
        bag_label = self.bag_labels[index]


        # Process the image for 'query' and 'key'
        image_data_q = self.tsfms(Image.open(img_path).convert("RGB"))
        image_data_k = self.tsfms(Image.open(img_path).convert("RGB"))

        return (image_data_q, image_data_k), bag_label

    
    def __len__(self):
        return len(self.unique_bag_ids)
    
    def n_features(self):
        return self.data.size(1)
    
    
def collate_custom(batch):
    batch_data_q = []  # List to store query images
    batch_data_k = []  # List to store key images
    batch_labels = []  # List to store bag labels

    for (image_data_q, image_data_k), bag_label in batch:
        batch_data_q.append(image_data_q)
        batch_data_k.append(image_data_k)
        batch_labels.append(bag_label)

    # Stack the images and labels
    batch_data_q = torch.stack(batch_data_q).cuda()
    batch_data_k = torch.stack(batch_data_k).cuda()
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()

    return (batch_data_q, batch_data_k), batch_labels

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
    


def train(loader, model, teacher, criterion, optimizer, epoch, args):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    
    for idx, (images, targets) in enumerate(loader):
        data_time.update(time.time() - end)
        warmup_learning_rate(args, epoch, idx, len(loader), optimizer)
        
        bsz = targets.shape[0]
        im_q, im_k = images
        if torch.cuda.is_available():
            im_q = im_q.cuda(non_blocking=True)
            im_k = im_k.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
        if args.mix: # image-based regularizations
            im_q, y0a, y0b, lam0 = mix_fn(im_q, targets, args.mix_alpha, args.mix)
            im_k, y1a, y1b, lam1 = mix_fn(im_k, targets, args.mix_alpha, args.mix)
            images = torch.cat([im_q, im_k], dim=0)
            l_q = mix_target(y0a, y0b, lam0, args.num_classes)
            l_k = mix_target(y1a, y1b, lam1, args.num_classes)
        else:
            images = torch.cat([im_q, im_k], dim=0)
            l_q = F.one_hot(targets, args.num_classes)
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



def save_model(model, optimizer, args, epoch, save_path, val_loss_best):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss_best': val_loss_best,
        'args': args
    }
    torch.save(state, save_path)


class Args:
    def __init__(self, warm, start_epoch, warm_epochs, learning_rate, lr_decay_rate, num_classes, epochs, warmup_from, cosine, lr_decay_epochs, mix, mix_alpha, KD_temp, KD_alpha, print_freq, teacher_path, teacher_ckpt):
        self.warm = warm
        self.start_epoch = start_epoch
        self.warm_epochs = warm_epochs
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.num_classes = num_classes
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
    model_name = 'GenSCL_1'
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
        start_epoch=0,
        warm_epochs=5,
        learning_rate=0.01,
        lr_decay_rate=0.1,
        num_classes = 2,
        epochs=50,
        warmup_from=0.001,
        cosine=True,
        lr_decay_epochs=[30, 40],
        mix='mixup',
        mix_alpha=0.2,
        KD_temp=4,
        KD_alpha=0.9,
        print_freq=100,
        teacher_path=model_folder,
        teacher_ckpt='teacher_ITS2CLR.pth'
    )

    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    #export_location = '/home/paperspace/cadbusi-LFS/export_09_28_2023/'
    #cropped_images = f"/home/paperspace/Temp_Data/{img_size}_images/"
    

    # Get Training Data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_labels = len(label_columns)

    print("Training Data...")
    #train_dataset = TUD.Subset(ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='all'),list(range(0,100)))
    #dataset_val = TUD.Subset(ITS2CLR_Dataset(bags_val, train=False, save_processed=False),list(range(0,100)))
    train_dataset = ITS2CLR_Dataset(bags_train, train=True, save_processed=False, bag_type='all')
    dataset_val = ITS2CLR_Dataset(bags_val, train=False)

            
    # Create data loaders
    train_dl =  TUD.DataLoader(train_dataset, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)

    
    # Create model
    model = SupConResNet(name='resnet18').cuda()
    criterion = GenSupConLoss(temperature=0.07)
    val_criterion = nn.BCELoss()
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
    os.makedirs(model_folder, exist_ok=True)
    student_path = f'{model_folder}/{model_name}.pth'
    if os.path.exists(student_path):
        ckpt = torch.load(student_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

        if 'optimizer' in ckpt and 'epoch' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
            args.start_epoch = ckpt['epoch'] + 1

        if 'loss_best' in ckpt:
            loss_best = ckpt['loss_best']
        else:
            loss_best = 99999  # Default value if not found in the checkpoint

        print(f"=> Successfully loaded checkpoint '{student_path}' at epoch {args.start_epoch}")
    else:
        print(f"{model_name} does not exist, creating a new instance")
        loss_best = 99999
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")
        

    # Training loop
    #init_wandb(args)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        res = train(train_dl, model, teacher, criterion, optimizer, epoch, args)
        time2 = time.time()
        #print(f'epoch {epoch}, total time {format_time(time2 - time1)}')
        #wandb.log(res, step=epoch)
        
        loss = res['trn_loss']
        
        if loss < loss_best:
            loss_best = loss
            save_model(model, optimizer, args, epoch, student_path, loss_best)
            print("Saved checkpoint due to improved loss")
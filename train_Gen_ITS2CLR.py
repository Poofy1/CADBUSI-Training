import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from archs.save_arch import *
from data.ITS2CLR_util import *
from torch.optim import Adam
from data.format_data import *
from archs.model_GenSCL import *
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    
    
class Bag_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, transform=None, save_processed=False, bag_type='all'):
        self.bags_dict = bags_dict
        self.save_processed = save_processed
        self.transform = transform
        self.unique_bag_ids = list(bags_dict.keys())

        # Filter bags based on bag_type
        if bag_type != 'all':
            self.unique_bag_ids = [bag_id for bag_id in self.unique_bag_ids
                                if (self.bags_dict[bag_id]['bag_labels'][0] == 1 and bag_type == 'positive') or 
                                    (self.bags_dict[bag_id]['bag_labels'][0] == 0 and bag_type == 'negative')]
    
    def __getitem__(self, index):
        actual_id = self.unique_bag_ids[index]
        bag_info = self.bags_dict[actual_id]

        # Extract labels, image file paths, and instance-level labels
        bag_labels = bag_info['bag_labels']
        files_this_bag = bag_info['images']
        instance_labels = bag_info['image_labels']

        # Process images
        image_data = torch.stack([self.transform(Image.open(fn).convert("RGB")) for fn in files_this_bag])
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

    
class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, transform=None, save_processed=False):
        self.bags_dict = bags_dict
        self.save_processed = save_processed
        self.transform = transform
        self.unique_bag_ids = list(bags_dict.keys())
        
        self.images = []  # List to store individual images
        self.bag_labels = []  # List to store corresponding bag labels for each image

        # Iterate over each bag and add images and labels to the lists
        for bag_id in self.unique_bag_ids:
            bag_info = bags_dict[bag_id]
            for img_path in bag_info['images']:
                self.images.append(img_path)
                self.bag_labels.append(bag_info['bag_labels'][0])

    def __getitem__(self, index):
        img_path = self.images[index]
        bag_label = self.bag_labels[index]

        # Process the image for 'query' and 'key'
        image_data_q = self.transform(Image.open(img_path).convert("RGB"))
        image_data_k = self.transform(Image.open(img_path).convert("RGB"))

        # Create a boolean value that is True if bag_label is False, and vice versa
        label_bool = not bool(bag_label)

        return (image_data_q, image_data_k), bag_label, label_bool

    
    def __len__(self):
        return len(self.unique_bag_ids)
    
    def n_features(self):
        return self.data.size(1)
    
    
    
def collate_instance(batch):
    batch_data_q = []  # List to store query images
    batch_data_k = []  # List to store key images
    batch_labels = []  # List to store bag labels
    batch_label_bools = []  # List to store the boolean values

    for (image_data_q, image_data_k), bag_label, label_bool in batch:
        batch_data_q.append(image_data_q)
        batch_data_k.append(image_data_k)
        batch_labels.append(bag_label)
        batch_label_bools.append(label_bool)

    # Stack the images and labels
    batch_data_q = torch.stack(batch_data_q).cuda()
    batch_data_k = torch.stack(batch_data_k).cuda()
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
    batch_label_bools = torch.tensor(batch_label_bools, dtype=torch.bool).cuda()

    return (batch_data_q, batch_data_k), batch_labels, batch_label_bools


def collate_bag(batch):
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


def spl_scheduler(current_epoch, total_epochs, initial_ratio, final_ratio):
    if current_epoch < warmup_epochs:
        return initial_ratio
    else:
        return initial_ratio + (final_ratio - initial_ratio) * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)


def default_train():
    global val_loss_best

    # Training phase
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for (data, yb, instance_yb, id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
        num_bags = len(data)
        all_images = torch.cat(data, dim=0).cuda()
        yb = yb.cuda()

        optimizer.zero_grad()

        # Process all images and then split per bag
        features, logits = model(all_images)
        split_sizes = [bag.size(0) for bag in data]
        logits_per_bag = torch.split(logits, split_sizes, dim=0)

        batch_loss = 0.0
        for i, logit in enumerate(logits_per_bag):
            bag_logit = logit.sum(dim=0, keepdim=False)
            bag_logit = torch.sigmoid(bag_logit)
            loss = loss_func(bag_logit, yb[i])
            batch_loss += loss
            predicted = (torch.sigmoid(bag_logit) > 0.5).float()
            correct += (predicted == yb[i]).sum().item()

        batch_loss /= num_bags
        total_loss += batch_loss.item() 
        total += num_bags  # Assuming one label per bag

        batch_loss.backward()
        optimizer.step()

    train_loss = total_loss / len(bag_dataloader_train)
    train_acc = correct / total

    # Evaluation phase
    model.eval()
    total_val_loss = 0.0
    total = 0
    correct = 0
    all_targs = []
    all_preds = []

    with torch.no_grad():
        for (data, yb, instance_yb, id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)):
            num_bags = len(data)
            all_images = torch.cat(data, dim=0).cuda()
            yb = yb.cuda()

            # Process all images and then split per bag
            features, logits = model(all_images)
            split_sizes = [bag.size(0) for bag in data]
            logits_per_bag = torch.split(logits, split_sizes, dim=0)

            batch_loss = 0.0
            for i, logit in enumerate(logits_per_bag):
                bag_logit = logit.sum(dim=0, keepdim=False)
                bag_logit = torch.sigmoid(bag_logit)
                loss = loss_func(bag_logit, yb[i])
                batch_loss += loss
                predicted = (bag_logit > 0.5).float()  # Prediction based on the summed logits
                correct += (predicted == yb[i]).sum().item()

                #confusion_matrix
                target_scalar = yb[i].item() 
                predicted_list = predicted.squeeze().tolist() 
                all_targs.append(target_scalar)
                all_preds.append(predicted_list)

            batch_loss /= num_bags
            total_val_loss += batch_loss.item()
            total += num_bags

    val_loss = total_val_loss / len(bag_dataloader_val)
    val_acc = correct / total

    train_losses_over_epochs.append(train_loss)
    valid_losses_over_epochs.append(val_loss)
    
    print(f"Epoch {epoch+1} | Acc   | Loss")
    print(f"Train   | {train_acc:.4f} | {train_loss:.4f}")
    print(f"Val     | {val_acc:.4f} | {val_loss:.4f}")
        
    # Save the model
    if val_loss < val_loss_best:
        val_loss_best = val_loss  # Update the best validation accuracy
        save_state(epoch, label_columns, train_acc, val_loss, val_acc, model_folder, model_name, model, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs)
        print("Saved checkpoint due to improved val_loss")


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
    model_name = 'Gen_ITS2CLR_test'
    encoder_arch = 'resnet18'
    dataset_name = 'export_12_26_2023'
    label_columns = ['Has_Malignant']
    instance_columns = [] #['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 350
    bag_batch_size = 2
    min_bag_size = 2
    max_bag_size = 10
    instance_batch_size = 16
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
    
    #ITS2CLR Config
    feature_extractor_train_count = 10
    initial_ratio = 1 #100% negitive bags
    final_ratio = 0.7  #20% negitive bags
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
    
    
    
    train_transform = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.05), scale=(1, 1.2),),
                CLAHETransform(),
                T.ToTensor(),
                GaussianNoise(mean=0, std=0.015),  # Add slight noise
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    val_transform = T.Compose([
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    print("Training Data...")
    # Create datasets
    bag_dataset_train = TUD.Subset(Bag_Dataset(bags_train, transform=train_transform, save_processed=False),list(range(0,100)))
    bag_dataset_val = TUD.Subset(Bag_Dataset(bags_val, transform=val_transform, save_processed=False),list(range(0,100)))
    instance_dataset_train = TUD.Subset(Instance_Dataset(bags_train, transform=train_transform, save_processed=False),list(range(0,100)))
    instance_dataset_val = TUD.Subset(Instance_Dataset(bags_val, transform=val_transform, save_processed=False),list(range(0,100)))
    
    #bag_dataset_train = ITS2CLR_Dataset(bags_train, transform=train_transform, save_processed=False)
    #bag_dataset_val = ITS2CLR_Dataset(bags_val, transform=val_transform, save_processed=False)
    #instance_dataset_train = Instance_Dataset(bags_train, transform=train_transform, save_processed=False)
    #instance_dataset_val = Instance_Dataset(bags_val, transform=val_transform, save_processed=False)

            
    # Create data loaders
    bag_dataloader_train =  TUD.DataLoader(bag_dataset_train, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val =  TUD.DataLoader(bag_dataset_val, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True)
    instance_dataloader_train =  TUD.DataLoader(instance_dataset_train, batch_size=instance_batch_size, collate_fn = collate_instance, drop_last=True, shuffle = True)
    instance_dataloader_val =    TUD.DataLoader(instance_dataset_val, batch_size=instance_batch_size, collate_fn = collate_instance, drop_last=True)

    

    # Get Model
    model = SupConResNet_custom(name='resnet18', num_classes=1).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")        
        
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    loss_func = nn.BCELoss()
    genscl = GenSupConLoss(temperature=0.07)
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    epoch_start = 0
    

    
    # Check if the model already exists
    model_folder = f"{env}/models/{model_name}/"
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
        val_loss_best = 99999


    # Training loop
    for epoch in range(epoch_start, total_epochs):
        if epoch % (1 + feature_extractor_train_count) == 0:
            
            print('Training Default')
            default_train()
        else:
            # Generalized Supervised Contrastive Learning phase
            print('Training Feature Extractor')
            model.train()
            losses = AverageMeter()
            
            # Get difficualy ratio
            current_ratio = spl_scheduler(epoch, total_epochs, initial_ratio, final_ratio)
            print(f'Current Bag Ratio: {current_ratio:.2f}')
            pos_batch_size = int(current_ratio * instance_batch_size)
            neg_batch_size = instance_batch_size - pos_batch_size
            
            epoch_loss = 0.0

            # Iterate over the training data
            for idx, (images, batch_labels, batch_label_bools) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                warmup_learning_rate(args, epoch, idx, len(instance_dataloader_train), optimizer)
                
                bsz = batch_labels.shape[0]
                im_q, im_k = images
                if torch.cuda.is_available():
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    batch_labels = batch_labels.cuda(non_blocking=True)
                    
                if args.mix: # image-based regularizations
                    im_q, y0a, y0b, lam0 = mix_fn(im_q, batch_labels, args.mix_alpha, args.mix)
                    im_k, y1a, y1b, lam1 = mix_fn(im_k, batch_labels, args.mix_alpha, args.mix)
                    images = torch.cat([im_q, im_k], dim=0)
                    l_q = mix_target(y0a, y0b, lam0, args.num_classes)
                    l_k = mix_target(y1a, y1b, lam1, args.num_classes)
                else:
                    images = torch.cat([im_q, im_k], dim=0)
                    l_q = F.one_hot(batch_labels, args.num_classes)
                    l_k = l_q
                

                # forward
                features, pred = model(images)
                features = F.normalize(features, dim=1)
                features = torch.split(features, [bsz, bsz], dim=0)
                
                # no KD
                loss = genscl(features, [l_q, l_k])
                
                losses.update(loss.item(), bsz)
                # backwaqrd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f'loss {loss:.3f}')

                
                
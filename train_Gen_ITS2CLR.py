import os, pickle, random
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
        self.image_labels = []  # List to store individual image labels

        # Iterate over each bag and add images, bag labels, and image labels to the lists
        for bag_id in self.unique_bag_ids:
            bag_info = bags_dict[bag_id]
            for idx, img_path in enumerate(bag_info['images']):
                self.images.append(img_path)
                self.bag_labels.append(bag_info['bag_labels'][0])
                self.image_labels.append(bag_info['image_labels'][idx])

        self.positive_indices = [i for i, label in enumerate(self.bag_labels) if label]
        self.negative_indices = [i for i, label in enumerate(self.bag_labels) if not label]


    def __getitem__(self, index):
        img_path = self.images[index]
        bag_label = self.bag_labels[index]
        image_label = self.image_labels[index]

        # Use transform on the image
        image_data_q = self.transform(Image.open(img_path).convert("RGB"))
        image_data_k = self.transform(Image.open(img_path).convert("RGB"))
        
        # Determine label and confidence based on image_label presence
        if image_label != [None]:  # This implies specific labels exist for the image
            instance_label = image_label[0] # Use the specific image label(s)
            label_confidence = 1  # Set confidence to 1 for specific labels
        else:
            instance_label = bag_label  # Use the bag label as a fallback
            label_confidence = int(not instance_label)  # Confidence based on bag label presence

        return (image_data_q, image_data_k), instance_label, label_confidence
    
    
    def __len__(self):
        return len(self.images) 
    
    # Utility function to get positive and negative indices
    def get_pos_neg_indices(self):
        return self.positive_indices, self.negative_indices
    
class BalancedInstanceSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_indices, self.negative_indices = dataset.get_pos_neg_indices()
        self.total_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.total_batches):
            batch = random.sample(self.positive_indices, 2) + random.sample(self.negative_indices, 2)
            while len(batch) < self.batch_size:
                batch.append(random.choice(self.positive_indices + self.negative_indices))
            random.shuffle(batch)
            yield batch  # Yield the entire batch as a list

    def __len__(self):
        return self.total_batches
    
    
def collate_instance(batch):
    batch_data_q = []
    batch_data_k = [] 
    batch_labels = []
    batch_label_confidence = [] 

    for (image_data_q, image_data_k), bag_label, label_bool in batch:
        batch_data_q.append(image_data_q)
        batch_data_k.append(image_data_k)
        batch_labels.append(bag_label)
        batch_label_confidence.append(label_bool)

    # Stack the images and labels
    batch_data_q = torch.stack(batch_data_q).cuda()
    batch_data_k = torch.stack(batch_data_k).cuda()
    batch_labels = torch.tensor(batch_labels, dtype=torch.long).cuda()
    batch_label_confidence = torch.tensor(batch_label_confidence, dtype=torch.int).cuda()

    return (batch_data_q, batch_data_k), batch_labels, batch_label_confidence


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
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(GenSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, mapped_anchors):
        '''
        Args:
            features: (anchor_features, contrast_features), each: [N, feat_dim]
            labels: (anchor_labels, contrast_labels) each: [N, num_cls]
            mapped_anchors: [N,], boolean tensor indicating anchor status
        '''
        if self.contrast_mode == 'dynamic':
            # features is a tuple (anchor_features, contrast_features)
            anchor_features_map, contrast_features_map = features
            
            # Ensure mapped_anchors is a boolean tensor for indexing
            anchor_indices = mapped_anchors.nonzero(as_tuple=True)[0]
            contrast_indices = (~mapped_anchors).nonzero(as_tuple=True)[0]
            
            # Now select the features and labels based on anchor and contrast indices
            anchor_features = anchor_features_map[anchor_indices]
            contrast_features = contrast_features_map[contrast_indices]

            anchor_labels = labels[0][anchor_indices].float()
            contrast_labels = labels[1][contrast_indices].float()
            
        elif self.contrast_mode == 'all': # anchor+contrast @ anchor+contrast
            anchor_labels = torch.cat(labels, dim=0).float()
            contrast_labels = anchor_labels
            
            anchor_features = torch.cat(features, dim=0)
            contrast_features = anchor_features
        elif self.contrast_mode == 'one': # anchor @ contrast
            anchor_labels = labels[0].float()
            contrast_labels = labels[1].float()
            
            anchor_features = features[0]
            contrast_features = features[1]
        
        print(anchor_labels)
        print(contrast_labels)
        
        # 1. compute similarities among targets
        anchor_norm = torch.norm(anchor_labels, p=2, dim=-1, keepdim=True) # [anchor_N, 1]
        contrast_norm = torch.norm(contrast_labels, p=2, dim=-1, keepdim=True) # [contrast_N, 1]
        
        print(anchor_norm.shape)
        print(contrast_norm.shape)
        
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
    



def prediction_anchor_scheduler(current_epoch, total_epochs, warmup_epochs, initial_ratio, final_ratio):
    if current_epoch < warmup_epochs:
        return initial_ratio
    else:
        return initial_ratio + (final_ratio - initial_ratio) * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    



def anchor_selection(model_preds, number_of_best, mixed_confidence_q, mixed_confidence_k):
    # Assuming model_preds is a tensor of shape [N, 1] where N is even
    model_preds_flattened = model_preds.view(-1)
    
    # Reshape to [-1, 2] to prepare for pair-wise multiplication, assuming N is even
    model_preds_reshaped = model_preds_flattened.view(-1, 2)
    
    # Multiply pairs: [0]*[1], [2]*[3], etc., by multiplying elements in the reshaped tensor
    model_preds_multiplied = model_preds_reshaped[:, 0] * model_preds_reshaped[:, 1]
    
    # Now find the adjusted version (distance from 0.5)
    model_preds_adjusted = torch.abs(model_preds_multiplied - 0.5)

    # Element-wise multiplication of mixed_confidence_q and mixed_confidence_k
    default_confidence = mixed_confidence_q * mixed_confidence_k
    
    # Initialize 'anchors' to zeros initially
    anchors = torch.zeros_like(default_confidence).int()

    # Select the top `number_of_best` predictions from the model and mark them as anchors
    _, top_indices = torch.topk(model_preds_adjusted, number_of_best, largest=True)
    anchors[top_indices] = 1

    # Apply default confidence threshold ONLY for those not already marked as anchors
    additional_anchors = (default_confidence >= 0.9) & (anchors == 0)
    anchors[additional_anchors] = 1
    
    # Ensure at least 2 positive and 2 negative anchors
    pos_indices = (anchors == 1).nonzero(as_tuple=True)[0]
    neg_indices = (anchors == 0).nonzero(as_tuple=True)[0]

    
    # Randomly flip positive anchors to negitive anchors
    if len(neg_indices) < 2:
        # Calculate how many anchors need to be flipped
        flips_needed = 2 - len(neg_indices)
        # Ensure not to reduce pos_indices below 2
        max_flips_possible = max(0, len(pos_indices) - 2)
        flips_to_perform = min(flips_needed, max_flips_possible)
        
        if flips_to_perform > 0:
            # Convert pos_indices tensor to a list for random sampling
            pos_indices_list = pos_indices.tolist()
            to_flip = random.sample(pos_indices_list, flips_to_perform)
            anchors[torch.tensor(to_flip, dtype=torch.long)] = 0
            
    # Use model pred to select more anchors if needed
    if len(pos_indices) < 2:
        additional_pos_needed = 2 - len(pos_indices)
        not_selected = (anchors == 0)  # Indices not already selected as anchors
        # Re-select from the top model predictions that are not yet selected as anchors
        _, additional_top_indices = torch.topk(model_preds_adjusted[not_selected], additional_pos_needed, largest=True)
        # Convert these indices to the original indexing scheme
        original_indices = torch.nonzero(not_selected).view(-1)[additional_top_indices]
        anchors[original_indices] = 1


    print("Default confidence:\n", default_confidence)
    print("Anchors:\n", anchors)

    return anchors



    





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
            # Find the max sigmoid value for each bag
            bag_max_output = logit.max(dim=0)[0]
            loss = loss_func(bag_max_output, yb[i])
            batch_loss += loss
            predicted = (bag_max_output > 0.5).float()
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
                # Find the max sigmoid value for each bag
                bag_max_output = logit.max(dim=0)[0]
                loss = loss_func(bag_max_output, yb[i])
                batch_loss += loss
                predicted = (bag_max_output > 0.5).float()
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
    model_name = 'Gen_ITS2CLR_test_2'
    encoder_arch = 'resnet18'
    dataset_name = 'export_01_31_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Reject Image', 'Malignant Lesion Present']   #['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 350
    bag_batch_size = 2
    min_bag_size = 2
    max_bag_size = 10
    instance_batch_size = 8
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
    initial_ratio = 0 # 0% preditions included
    final_ratio = 0.7 # 70% preditions included
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
    #bag_dataset_train = TUD.Subset(Bag_Dataset(bags_train, transform=train_transform, save_processed=False),list(range(0,100)))
    #bag_dataset_val = TUD.Subset(Bag_Dataset(bags_val, transform=val_transform, save_processed=False),list(range(0,100)))
    #instance_dataset_train = TUD.Subset(Instance_Dataset(bags_train, transform=train_transform, save_processed=False),list(range(0,100)))

    
    bag_dataset_train = Bag_Dataset(bags_train, transform=train_transform, save_processed=False)
    bag_dataset_val = Bag_Dataset(bags_val, transform=val_transform, save_processed=False)
    instance_dataset_train = Instance_Dataset(bags_train, transform=train_transform, save_processed=False)
    
    instance_sampler = BalancedInstanceSampler(instance_dataset_train, batch_size=instance_batch_size) 
            
    # Create data loaders
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True)
    instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=instance_sampler, collate_fn = collate_instance)
    

    # Get Model
    model = SupConResNet_custom(name=encoder_arch, num_classes=num_labels).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")        
        
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    loss_func = nn.BCELoss()
    genscl = GenSupConLoss(temperature=0.07, contrast_mode='dynamic')
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
            
            # When Bag AUC increases goto Feature extractor
            
            # Used the instance predictions from bag training to update the Instance Dataloader
            
        else:
            # Generalized Supervised Contrastive Learning phase
            print('Training Feature Extractor')
            model.train()
            losses = AverageMeter()
            
            # Get difficualy ratio
            predictions_ratio = prediction_anchor_scheduler(epoch, total_epochs, warmup_epochs, initial_ratio, final_ratio)
            predictions_included = round(predictions_ratio * instance_batch_size)
            print(f'Including Predictions: {predictions_ratio:.2f} ({predictions_included})')

            # Predictions ratio is not per batch, for the entire dataset
            # Update instance dataloader, only include confident instances or negitive bags or instance labels
            
            
            epoch_loss = 0.0
            total_loss = 0

            # Iterate over the training data
            for idx, (images, instance_labels, _) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                warmup_learning_rate(args, epoch, idx, len(instance_dataloader_train), optimizer)
                
                # Data preparation 
                bsz = instance_labels.shape[0]
                im_q, im_k = images
                im_q = im_q.cuda(non_blocking=True)
                im_k = im_k.cuda(non_blocking=True)
                instance_labels = instance_labels.cuda(non_blocking=True)
                
                # image-based regularizations (mixup)
                # lam 1 = no mixup
                im_q, y0a, y0b, lam0 = mix_fn(im_q, instance_labels, args.mix_alpha, args.mix)
                im_k, y1a, y1b, lam1 = mix_fn(im_k, instance_labels, args.mix_alpha, args.mix)
                images = torch.cat([im_q, im_k], dim=0)
                l_q = mix_target(y0a, y0b, lam0, args.num_classes)
                l_k = mix_target(y1a, y1b, lam1, args.num_classes)

                # forward
                features, pred = model(images)
                features = F.normalize(features, dim=1)
                features = torch.split(features, [bsz, bsz], dim=0)
                
                # Get anchors
                mapped_anchors = anchor_selection(pred, predictions_included, mixed_confidence_q, mixed_confidence_k)
                # In warmup use the whole batch and then Only true labels (no model predictions) as anchors 
                # After warmup the whole batch is an anchor contrasted against the whole batch
                # Anchors are on Instance level
                
                
                mapped_anchors = torch.tensor(mapped_anchors)
                
                # get loss (no teacher)
                loss = genscl(features, [l_q, l_k], mapped_anchors)
                losses.update(loss.item(), bsz)
                
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            #print(f'loss: {losses:.3f}')

                
                
import os, pickle
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
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=False):
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
        
        #img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads in BGR, so convert to RGB
        img = Image.fromarray(img)
        image_data = self.transform(img)

        return image_data, instance_label


    def __len__(self):
        return len(self.images)


class WarmupSampler(Sampler):
    def __init__(self, dataset, batch_size, strategy=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.indices_0 = [i for i, label in enumerate(self.dataset.final_labels) if label == 0]
        self.indices_1 = [i for i, label in enumerate(self.dataset.final_labels) if label == 1]
        self.indices_unlabeled = [i for i, label in enumerate(self.dataset.final_labels) if label == -1]

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size

        for _ in range(total_batches):
            if self.strategy == 1:
                # Ensure at least one positive sample
                num_positives = random.randint(1, max(1, min(len(self.indices_1), self.batch_size - 1)))
                num_negatives = self.batch_size - num_positives
                batch_positives = random.sample(self.indices_1, num_positives)
                batch_negatives = random.sample(self.indices_0, num_negatives)
                batch = batch_positives + batch_negatives
            elif self.strategy == 2:
                # Aim for 50/50 balance
                num_positives = self.batch_size // 2
                num_negatives = self.batch_size - num_positives
                batch_positives = random.sample(self.indices_1, num_positives)
                batch_negatives = random.sample(self.indices_0, num_negatives)
                batch = batch_positives + batch_negatives
            elif self.strategy == 3:
                # At least one positive, one negative, and possibility of -1 labels
                num_positives = max(1, random.randint(1, self.batch_size // 2))
                num_negatives = max(1, random.randint(1, self.batch_size // 2))
                num_unlabeled = self.batch_size - num_positives - num_negatives

                batch_positives = random.sample(self.indices_1, num_positives)
                batch_negatives = random.sample(self.indices_0, num_negatives)
                
                if num_unlabeled > 0 and self.indices_unlabeled:
                    batch_unlabeled = random.sample(self.indices_unlabeled, min(num_unlabeled, len(self.indices_unlabeled)))
                    batch = batch_positives + batch_negatives + batch_unlabeled
                else:
                    # If no unlabeled samples or not enough space, fill with more labeled samples
                    remaining = self.batch_size - len(batch_positives) - len(batch_negatives)
                    additional_samples = random.sample(self.indices_0 + self.indices_1, remaining)
                    batch = batch_positives + batch_negatives + additional_samples
            else:
                raise ValueError("Invalid strategy. Choose 1, 2, or 3")

            random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return len(self.dataset) // self.batch_size
    
    
def collate_instance(batch):
    batch_data = []
    batch_labels = []

    for image_data, bag_label in batch:
        batch_data.append(image_data)
        batch_labels.append(bag_label)

    # Stack the images and labels
    batch_data = torch.stack(batch_data)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return batch_data, batch_labels




class PALM(nn.Module):
    def __init__(self, args, num_classes=2, n_protos=100, proto_m=0.99, temp=0.1, lambda_pcon=1, k=5, feat_dim=128, epsilon=0.05, unlabeled_weight=0.5):
        super(PALM, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  # temperature scaling
        self.nviews = args.nviews
        self.cache_size = args.cache_size
        self.unlabeled_weight = unlabeled_weight
        
        self.lambda_pcon = lambda_pcon
        
        self.feat_dim = feat_dim
        
        self.epsilon = epsilon
        self.sinkhorn_iterations = 3
        self.k = min(k, self.cache_size)
        
        self.n_protos = n_protos
        self.proto_m = proto_m
        self.register_buffer("protos", torch.rand(self.n_protos,feat_dim))
        self.protos = F.normalize(self.protos, dim=-1)
        
        
        
        # Initialize class counts for each prototype
        self.proto_class_counts = torch.zeros(self.n_protos, self.num_classes).cuda() # ADDED
        
    def sinkhorn(self, features):
        out = torch.matmul(features, self.protos.detach().T)
            
        Q = torch.exp(out.detach() / self.epsilon).t()# Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if torch.isinf(sum_Q):
            self.protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, self.ws(self.protos.detach()).T)
            Q = torch.exp(out.detach() / self.epsilon).t()# Q is K-by-B for consistency with notations from our paper
            sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q = F.normalize(Q, dim=1, p=1)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q = F.normalize(Q, dim=0, p=1)
            Q /= B

        Q *= B
        return Q.t()
        
    def mle_loss(self, features, targets):
        # update prototypes by EMA
        #features = torch.cat(torch.unbind(features, dim=1), dim=0)
        # Disabled becuase features are already correct shape?
        
        anchor_labels = targets.contiguous().repeat(self.nviews).view(-1, 1)
        contrast_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda()
                
        Q = self.sinkhorn(features)

        # topk
        if self.k > 0:
            update_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            update_mask = F.normalize(F.normalize(topk_mask*update_mask, dim=1, p=1),dim=0, p=1)
        # original
        else:
            update_mask = F.normalize(F.normalize(mask * Q, dim=1, p=1),dim=0, p=1)
        update_features = torch.matmul(update_mask.T, features)
        
        self.proto_class_counts += torch.matmul(update_mask.T, F.one_hot(targets, num_classes=self.num_classes).float()) # ADDED
        
        protos = self.protos
        protos = self.proto_m * protos + (1-self.proto_m) * update_features

        self.protos = F.normalize(protos, dim=1, p=2)
        
        Q = self.sinkhorn(features)
        
        proto_dis = torch.matmul(features, self.protos.detach().T)
        anchor_dot_contrast = torch.div(proto_dis, self.temp)
        logits = anchor_dot_contrast
       
        if self.k > 0:
            loss_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            loss_mask = F.normalize(topk_mask*loss_mask, dim=1, p=1)
            masked_logits = loss_mask * logits 
        else:  
            masked_logits = F.normalize(Q*mask, dim=1, p=1) * logits
    
        pos=torch.sum(masked_logits, dim=1)
        neg=torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
        log_prob=pos-neg
        
        loss = -torch.mean(log_prob)
        return loss   
    
    def proto_contra(self):
        
        protos = F.normalize(self.protos, dim=1)
        batch_size = self.num_classes
        
        proto_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(proto_labels, proto_labels.T).float().cuda()    

        contrast_count = self.cache_size
        contrast_feature = protos

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            0.5)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to('cuda'),
            0
        )
        mask = mask*logits_mask
        
        pos = torch.sum(F.normalize(mask, dim=1, p=1)*logits, dim=1)
        neg=torch.log(torch.sum(logits_mask * torch.exp(logits), dim=1))
        log_prob=pos-neg

        # loss
        loss = - torch.mean(log_prob)
        return loss
    
    
    def predict(self, features):
        # Assign the majority class to each prototype based on class counts
        _, proto_classes = torch.max(self.proto_class_counts, dim=1)
        
        # Compute the similarity between input features and prototypes
        similarity = torch.matmul(features, self.protos.T)
        
        # Get the index of the prototype with the highest similarity
        _, prototype_indices = torch.max(similarity, dim=1)
        
        # Map the prototype indices to their corresponding class labels
        predicted_classes = proto_classes[prototype_indices]
        
        return predicted_classes
           
    def unlabeled_loss(self, features):
        # Compute similarities between features and prototypes
        similarities = torch.matmul(features, self.protos.T)
        
        # Find the closest prototype for each feature
        closest_proto_idx = torch.argmax(similarities, dim=1)
        
        # Compute the loss as 1 minus cosine similarity to the closest prototype
        closest_similarities = similarities[torch.arange(features.size(0)), closest_proto_idx]
        loss = 1 - closest_similarities
        
        return loss.mean()  # Average loss per feature

    def forward(self, features, targets, unlabeled_features=None):
        loss = 0
        loss_dict = {}

        # Labeled data loss
        g_con = self.mle_loss(features, targets)
        loss += g_con
        loss_dict['mle'] = g_con.cpu().item()

        if self.lambda_pcon > 0:
            g_dis = self.lambda_pcon * self.proto_contra()
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()

        # Unlabeled data loss
        if unlabeled_features is not None and unlabeled_features.numel() > 0:
            u_loss = self.unlabeled_weight * self.unlabeled_loss(unlabeled_features)
            loss += u_loss
            #print(u_loss)
            loss_dict['unlabeled'] = u_loss.cpu().item()
            
        
        self.protos = self.protos.detach()
        
        return loss, loss_dict
    
    



if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "Palm6_CASBUSI_224"
    
    dataset_name = 'export_oneLesions' #'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  
    img_size = 224
    bag_batch_size = 3
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  50
    arch = 'efficientnet_b0'
    pretrained_arch = False

    
    """dataset_name = 'imagenette2_hard'
    label_columns = ['Has_Fish']
    instance_columns = ['Has_Fish']  
    img_size = 128
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    arch = 'resnet18'
    pretrained_arch = False"""

    #ITS2CLR Config
    feature_extractor_train_count = 6 # 6
    MIL_train_count = 6
    total_epochs = 50
    warmup_epochs = 10
    learning_rate=0.001
    reset_aggregator = True # Reset the model.aggregator weights after contrastive learning
    
    
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
    
    # LOSS INIT
    class Args:
        def __init__(self, nviews, cache_size):
            self.nviews = nviews
            self.cache_size = cache_size
    palm_args = Args(nviews=1, cache_size=50)
    palm = PALM(palm_args).cuda()
    BCE_loss = nn.BCELoss()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001
    
    
    # MODEL INIT
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
        "total_epochs": total_epochs,
        "reset_aggregator": reset_aggregator,
        "warmup_epochs": warmup_epochs,
        "learning_rate": learning_rate,
    }

    (model, optimizer, head_folder, pretrained_name, 
    model_folder, model_name, train_losses, valid_losses, epoch,
    val_acc_best, val_loss_best, selection_mask, 
    warmup, pickup_warmup) = setup_model(model, optimizer, config)

    #pickup_warmup = False
    warmup = true
    
    # Training loop
    while epoch < total_epochs:
        
        
        if not pickup_warmup: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, selection_mask, transform=train_transform, warmup=warmup)
            instance_dataset_val = Instance_Dataset(bags_val, selection_mask, transform=val_transform, warmup=True)
            train_sampler = WarmupSampler(instance_dataset_train, instance_batch_size, strategy=3)
            val_sampler = WarmupSampler(instance_dataset_val, instance_batch_size, strategy=1)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=4, collate_fn = collate_instance, pin_memory=True)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
            if warmup:
                target_count = warmup_epochs
            else:
                target_count = feature_extractor_train_count
            
            
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {warmup}')
            
            
            for i in range(target_count): 
                losses = AverageMeter()
                palm_total_correct = 0
                instance_total_correct = 0
                total_samples = 0
                model.train()
                
                # Iterate over the training data
                for idx, (images, instance_labels) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
  
                    # forward
                    optimizer.zero_grad()
                    _, _, instance_predictions, features = model(images, projector=True)
                    features.to(device)
                    
                    
                    # Separate labeled and unlabeled data
                    labeled_mask = instance_labels != -1
                    unlabeled_mask = instance_labels == -1
                    labeled_instance_labels = instance_labels[labeled_mask]
                    labeled_features = features[labeled_mask]
                    unlabeled_features = features[unlabeled_mask]
                    labeled_instance_predictions = instance_predictions[labeled_mask]
                    
                    # Get loss from PALM
                    palm_loss, loss_dict = palm(labeled_features, labeled_instance_labels, unlabeled_features)
                    #print(palm_loss)
                    
                    # Calculate BCE loss
                    bce_loss_value = BCE_loss(labeled_instance_predictions, labeled_instance_labels.float())
                    #print(bce_loss_value)

                    # Backward pass and optimization step
                    total_loss = palm_loss + bce_loss_value
                    total_loss.backward()
                    optimizer.step()
        
                    # Update the loss meter
                    losses.update(total_loss.item(), images[0].size(0))
                    
                    # Get predictions from PALM
                    with torch.no_grad():
                        palm_predicted_classes = palm.predict(labeled_features)
                        instance_predicted_classes = (labeled_instance_labels) > 0.5

                        # Calculate accuracy for PALM predictions
                        palm_correct = (palm_predicted_classes == labeled_instance_labels).sum().item()
                        palm_total_correct += palm_correct
                        
                        # Calculate accuracy for instance predictions
                        instance_correct = (instance_predicted_classes == labeled_instance_labels).sum().item()
                        instance_total_correct += instance_correct
                        
                        total_samples += labeled_instance_labels.size(0)

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
                        images = images.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)

                        # Forward pass
                        _, _, instance_predictions, features = model(images, projector=True)
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
                if instance_val_acc > val_acc_best:
                    val_acc_best = instance_val_acc
                    if warmup:
                        target_folder = head_folder
                        target_name = pretrained_name
                    else:
                        target_folder = model_folder
                        target_name = model_name
                    all_targs = []
                    all_preds = []
                    
                    
                    save_state(epoch, label_columns, instance_train_acc, 0, instance_val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, train_losses, valid_losses,)
                    print("Saved checkpoint due to improved val_acc")




        if pickup_warmup: 
            pickup_warmup = False
        if warmup:
            print("Warmup Phase Finished")
            warmup = False
            

        
            
        print('Predicting Missing Instances')
        for a in range(MIL_train_count):
            model.train()
            train_bag_logits = {}
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
                
            

            print(f"[{a+1}/{MIL_train_count}] | Acc | Loss")
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

                
                epoch += 1

import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
import scipy.stats
from tqdm import tqdm
import pickle
from torch import nn
from archs.save_arch import *
from data.Gen_ITS2CLR_util import *
import torch.optim as optim
from torch.utils.data import Sampler
from torch.optim import Adam
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from data.format_data import *
from archs.model_PALM import *
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    
class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, show_groundtruth = True):
        self.transform = transform

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
                
                # Only include confident instances (selection_mask) or negative bags or instance labels
                if label[0] is not None and show_groundtruth:
                    image_label = label[0]
                elif bag_label == 0:
                    image_label = 0
                elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                    image_label = selection_mask_labels[idx]
                
                if image_label is not None:
                    self.images.append(img)
                    self.final_labels.append(image_label)

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        
        img = Image.open(img_path).convert("RGB")
        image_data = self.transform(img)

        return image_data, instance_label


    def __len__(self):
        return len(self.images)


class InstanceSampler(Sampler):
    def __init__(self, dataset, batch_size, strategy=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.strategy = strategy
        self.indices_0 = [i for i, label in enumerate(self.dataset.final_labels) if label == 0]
        self.indices_1 = [i for i, label in enumerate(self.dataset.final_labels) if label == 1]

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size

        for _ in range(total_batches):
            if self.strategy == 1:
                # Ensure at least one positive sample
                num_positives = random.randint(1, max(1, min(len(self.indices_1), self.batch_size - 1)))
                num_negatives = self.batch_size - num_positives
            elif self.strategy == 2:
                # Aim for 50/50 balance
                num_positives = self.batch_size // 2
                num_negatives = self.batch_size - num_positives
            else:
                raise ValueError("Invalid strategy. Choose 1 or 2")

            batch_positives = random.sample(self.indices_1, num_positives)
            batch_negatives = random.sample(self.indices_0, num_negatives)

            batch = batch_positives + batch_negatives
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




class PALM(nn.Module):
    def __init__(self, args, num_classes=2, n_protos=100, proto_m=0.99, temp=0.1, lambda_pcon=1, k=5, feat_dim=128, epsilon=0.05):
        super(PALM, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  # temperature scaling
        self.nviews = args.nviews
        self.cache_size = args.cache_size
        
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
            
    def forward(self, features, targets):
        loss = 0
        loss_dict = {}

        g_con = self.mle_loss(features, targets)
        loss += g_con
        loss_dict['mle'] = g_con.cpu().item()
                    
        if self.lambda_pcon > 0:            
            g_dis = self.lambda_pcon * self.proto_contra()
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()
                                
        self.protos = self.protos.detach()
                
        return loss, loss_dict
    


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
        val_acc_best = saved_stats['val_loss']
    
    # Load the selection_mask dictionary from the file
    if os.path.exists(f'{target_folder}/selection_mask.pkl'):
        with open(f'{target_folder}/selection_mask.pkl', 'rb') as file:
            selection_mask = pickle.load(file)
            
    return train_losses, valid_losses, epoch, val_acc_best, selection_mask



if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "Palm3_noInstances"

    
    """dataset_name = 'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  
    img_size = 300
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  100
    use_efficient_net = False"""
    
    dataset_name = 'imagenette2'
    label_columns = ['Has_Fish']
    instance_columns = ['Has_Fish']  
    img_size = 128
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    use_efficient_net = False
    
    #ITS2CLR Config
    feature_extractor_train_count = 15 # 6
    MIL_train_count = 8
    initial_ratio = .3 #0.3 # --% preditions included
    final_ratio = 1 #0.85 # --% preditions included
    total_epochs = 20
    warmup_epochs = 30
    
    arch = "resnet18"
    pretrained_arch = False
    reset_aggregator = True # Reset the model.aggregator weights after contrastive learning
    
    learning_rate=0.001
    num_classes = len(label_columns) + 1

    
    # Get Training Data
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_labels = len(label_columns)
    
    """train_transform = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    val_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])"""
    
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


    # Create datasets
    #bag_dataset_train = TUD.Subset(BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False),list(range(0,100)))
    #bag_dataset_val = TUD.Subset(BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False),list(range(0,100)))
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
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
    
    
    class Args:
        def __init__(self, nviews, cache_size):
            self.nviews = nviews
            self.cache_size = cache_size
    palm_args = Args(nviews=1, cache_size=50)
    palm = PALM(palm_args, n_protos=100).cuda()
    BCE_loss = nn.BCELoss()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    

    model_name = f"{dataset_name}_{arch}_{model_version}"
    pretrained_name = f"Head_{head_name}_{arch}"
    
    head_folder = f"{env}/models/{pretrained_name}/"
    head_path = f"{head_folder}/{pretrained_name}.pth"
    head_optimizer_path = f"{head_folder}/{pretrained_name}_optimizer.pth"
    
    # Check if the model already exists
    model_folder = f"{env}/models/{pretrained_name}/{model_name}/"
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"
    
    val_acc_best = 0
    selection_mask = []
    epoch = 0
    warmup = False
    pickup_warmup = False
    train_losses = []
    valid_losses = []

    if os.path.exists(model_path): # If main model exists
        #model.load_state_dict(torch.load(model_path))
        #optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"Loaded pre-existing model from {model_name}")
        
        # Load only the encoder state dictionary
        encoder_state_dict = torch.load(model_path)
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_state_dict)

        train_losses, valid_losses, epoch, val_acc_best, selection_mask = load_state(stats_path, model_folder)
        
    else:
        print(f"{model_name} does not exist, creating new instance")

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
        }
        

        if os.path.exists(head_path):  # If main head model exists
            os.makedirs(model_folder, exist_ok=True)
            pickup_warmup = True
            #model.load_state_dict(torch.load(head_path))
            
            
            # Load only the encoder state dictionary
            encoder_state_dict = torch.load(head_path)
            encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_state_dict.items() if k.startswith('encoder.')}
            model.encoder.load_state_dict(encoder_state_dict)
            #optimizer.load_state_dict(torch.load(head_optimizer_path))
    
            print(f"Loaded pre-trained model from {pretrained_name}")
            
            config_path = f"{model_folder}/config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
                
            with open(f'{model_folder}model_architecture.txt', 'w') as f:
                print(model, file=f)
             
        else: # If main head model DOES NOT exists
            warmup = True
            os.makedirs(head_folder, exist_ok=True)
            os.makedirs(model_folder, exist_ok=True)

                
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
            instance_dataset_train = Instance_Dataset(bags_train, selection_mask, transform=train_transform, show_groundtruth = False)
            instance_dataset_val = Instance_Dataset(bags_val, selection_mask, transform=train_transform, show_groundtruth = True)
            #train_sampler = InstanceSampler(instance_dataset_train, instance_batch_size, strategy=1)
            val_sampler = InstanceSampler(instance_dataset_val, instance_batch_size, strategy=2)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_size=instance_batch_size, collate_fn = collate_instance)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
            if warmup:
                target_count = warmup_epochs
            else:
                target_count = feature_extractor_train_count
            
            
            
            print('Training Feature Extractor')
            

            model.train()
            for i in range(target_count): 
                losses = AverageMeter()
                total_correct = 0
                total_samples = 0
                
                # Iterate over the training data
                for idx, (images, instance_labels) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
  
                    # forward
                    optimizer.zero_grad()
                    _, _, features = model(images, projector=True)
                    features.to(device)
                    
                    # Get loss from PALM
                    loss, loss_dict = palm(features, instance_labels)

                    # Backward pass and optimization step
                    loss.backward()
                    optimizer.step()
        
                    # Update the loss meter
                    losses.update(loss.item(), images[0].size(0))
                    


                
                
                # Validation loop
                model.eval()
                val_losses = AverageMeter()
                val_total_correct = 0
                val_total_samples = 0

                # Collect features
                ftrain = []
                ftest = []

                with torch.no_grad():
                    ftrain = []
                    ftest = []
                    instance_labels_list = []  # New list to store instance labels

                    # Collect training features
                    for idx, (images, _) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                        images = images.cuda(non_blocking=True)
                        _, _, features = model(images, projector=True)
                        ftrain.append(features.cpu().numpy())

                    # Collect validation features and labels
                    for idx, (images, instance_labels) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        images = images.cuda(non_blocking=True)
                        _, _, features = model(images, projector=True)
                        ftest.append(features.cpu().numpy())
                        instance_labels_list.append(instance_labels.cpu().numpy())  # Save labels

                ftrain = np.concatenate(ftrain, axis=0)
                ftest = np.concatenate(ftest, axis=0)
                instance_labels = np.concatenate(instance_labels_list, axis=0)  # Concatenate all labels

                # Normalize features
                normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
                prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))
                ftrain = prepos_feat(ftrain)
                ftest = prepos_feat(ftest)

                mean_feat = ftrain.mean(0)
                std_feat = ftrain.std(0)
                prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
                ftrain_ssd = prepos_feat_ssd(ftrain)
                ftest_ssd = prepos_feat_ssd(ftest)

                cov = lambda x: np.cov(x.T, bias=True)

                def maha_score(X):
                    z = X - mean_feat
                    inv_sigma = np.linalg.pinv(cov(ftrain_ssd))
                    return -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)

                dtrain = maha_score(ftrain_ssd)  # Calculate scores for training data
                dtest = maha_score(ftest_ssd)    # Calculate scores for test data

                # Set threshold based on training data
                threshold = np.percentile(dtrain, 5)  # Assuming lower 5% of training scores are positive
                predicted_classes = (dtest <= threshold).astype(int)

                # Calculate accuracy
                correct = (predicted_classes == instance_labels).sum()
                total = len(instance_labels)
                val_acc = correct / total

                print(f'[{i+1}/{target_count}] Val Loss: N/A, Val Acc: {val_acc:.5f}')

                # Optional: print score distribution
                print(f"Training Mahalanobis scores range: {dtrain.min():.4f} to {dtrain.max():.4f}")
                print(f"Test Mahalanobis scores range: {dtest.min():.4f} to {dtest.max():.4f}")
                print(f"Threshold (based on training data): {threshold:.4f}")
                print(f"Shape of predicted_classes: {predicted_classes.shape}")
                print(f"Shape of instance_labels: {instance_labels.shape}")

                # Additional debugging information
                print(f"Number of positive predictions: {np.sum(predicted_classes)} / {len(predicted_classes)}")
                print(f"Number of positive labels: {np.sum(instance_labels)} / {len(instance_labels)}")
                                            
                """# Save the model
                if val_acc > val_acc_best:
                    val_acc_best = val_losses.avg
                    target_folder = head_folder
                    target_name = pretrained_name
                    all_targs = []
                    all_preds = []
                    
                    save_state(epoch, label_columns, train_acc, val_losses.avg, val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, train_losses, valid_losses,)
                    print("Saved checkpoint due to improved val_acc")"""
















        if pickup_warmup: 
            pickup_warmup = False
        if warmup:
            # Save the model and optimizer
            torch.save(model.state_dict(), head_path)
            torch.save(optimizer.state_dict(), head_optimizer_path)
            
            print("Warmup Phase Finished")
            warmup = False
        
        
        """print('Training Aggregator')   
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
                    #print(instance_pred)

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
            if val_loss < val_acc_best:
                val_acc_best = val_loss
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
                    pickle.dump(selection_mask, file)"""
                    
                    
                    
                    
                    
                    
            #exit() # TEMP DEBUGGING


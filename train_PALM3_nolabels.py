import os, pickle
from fastai.vision.all import *
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
from archs.model_PALM import *
from loss.palm import PALM
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

    


    


if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "Palm3_noInstances"

    
    """dataset_name = 'export_oneLesions' #'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  
    img_size = 300
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  50
    arch = 'resnet50'
    pretrained_arch = False
    """
    
    dataset_name = 'imagenette2_hard'
    label_columns = ['Has_Fish']
    instance_columns = ['Has_Fish']  
    img_size = 128
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    arch = 'resnet18'
    pretrained_arch = False

    
    #ITS2CLR Config
    feature_extractor_train_count = 6 # 6
    MIL_train_count = 6
    initial_ratio = .3 #0.3 # --% preditions included
    final_ratio = .8 #0.85 # --% preditions included
    total_epochs = 9999
    warmup_epochs = 20
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
    palm = PALM(nviews = 1, num_classes=2, n_protos=6, k = 5, lambda_pcon=1).cuda()
    BCE_loss = nn.BCELoss()
    
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    
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
        "initial_ratio": initial_ratio,
        "final_ratio": final_ratio,
        "total_epochs": total_epochs,
        "reset_aggregator": reset_aggregator,
        "warmup_epochs": warmup_epochs,
        "learning_rate": learning_rate,
    }

    model, optimizer, state = setup_model(model, optimizer, config)
    palm.load_state(state['palm_path'])
    
    # Training loop
    while state['epoch'] < total_epochs:
        
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            # Used the instance predictions from bag training to update the Instance Dataloader
            instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=False)
            instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True)
            #train_sampler = InstanceSampler(instance_dataset_train, instance_batch_size, strategy=1)
            val_sampler = InstanceSampler(instance_dataset_val, instance_batch_size, strategy=2)
            instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_size=instance_batch_size, collate_fn = collate_instance)
            instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
            
            if state['warmup']:
                target_count = warmup_epochs
            else:
                target_count = feature_extractor_train_count
            
            
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')
            

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
                                            
                """
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    if state['warmup']:
                        target_folder = state['head_folder']
                        target_name = state['pretrained_name']
                    else:
                        target_folder = state['model_folder']
                        target_name = state['model_name']
                    all_targs = []
                    all_preds = []
                    
                    
                    save_state(state['epoch'], label_columns, instance_train_acc, val_losses.avg, instance_val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                    palm.save_state(os.path.join(target_folder, "palm_state.pkl"), max_dist)
                    print("Saved checkpoint due to improved val_loss_instance")

                """



        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
        

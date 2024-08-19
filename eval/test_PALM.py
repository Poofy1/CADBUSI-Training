import torch
import os
import sys
import torch.utils.data as TUD
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from util.format_data import *
from util.sudo_labels import *
from loss.palm import PALM
from archs.save_arch import *
from archs.model_PALM2_solo import *

class Instance_Dataset_with_IDs(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=True):
        self.transform = transform
        self.warmup = warmup

        self.images = []
        self.final_labels = []
        self.bag_ids = []
        self.image_indices = []

        self.unique_bag_ids = list(bags_dict.keys())

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
                    # Return all images with unknown possibility 
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
                    self.bag_ids.append(bag_id_key)
                    self.image_indices.append(idx)

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        bag_id = self.bag_ids[index]
        image_index = self.image_indices[index]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads in BGR, so convert to RGB
        img = Image.fromarray(img)
        image_data = self.transform(img)

        return image_data, instance_label, bag_id, image_index
    
    def __len__(self):
        return len(self.images)

def collate_instance(batch):
    batch_data = []
    batch_labels = []
    batch_bag_ids = []
    batch_image_indices = []

    for image_data, bag_label, bag_id, image_index in batch:
        batch_data.append(image_data)
        batch_labels.append(bag_label)
        batch_bag_ids.append(bag_id)
        batch_image_indices.append(image_index)

    # Stack the images and labels
    batch_data = torch.stack(batch_data)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    return batch_data, batch_labels, batch_bag_ids, batch_image_indices


def test_model_and_collect_distances(model, palm, bag_dataloader, instance_dataloader, device):
    model.eval()
    
    # Bag-level metrics
    bag_targets, bag_predictions = [], []
    
    # Instance-level metrics
    instance_targets, fc_predictions, palm_predictions = [], [], []
    
    # Out-of-distribution counter
    ood_count = 0
    total_instances = 0
    instance_info = []
    
    # Collect distances
    distances = []
    instance_paths = []
    
    with torch.no_grad():
        # Bag-level testing
        for images, yb, _, _ in tqdm(bag_dataloader, desc="Testing bags"):
            bag_pred, _, _, _ = model(images, pred_on=True)
            bag_targets.extend(yb.cpu().numpy())
            bag_predictions.extend((bag_pred > 0.5).float().cpu().numpy())
        
        for images, instance_labels, bag_ids, image_indices in tqdm(instance_dataloader, desc="Testing instances"):
            images = images.to(device)
            _, _, fc_pred, features = model(images, projector=True)
            palm_pred, dist = palm.predict(features)
            
            distances.extend(dist.cpu().numpy())
            instance_info.extend(list(zip(bag_ids, image_indices)))
            
            # Check for out-of-distribution instances
            ood_instances = (dist > 1.4).sum().item()
            ood_count += ood_instances
            total_instances += len(instance_labels)
            
            instance_targets.extend(instance_labels.cpu().numpy())
            fc_predictions.extend((fc_pred > 0.5).float().cpu().numpy())
            palm_predictions.extend(palm_pred.cpu().numpy())
    
    ood_percentage = (ood_count / total_instances) * 100
    
    return (np.array(bag_targets), np.array(bag_predictions), 
            np.array(instance_targets), np.array(fc_predictions), np.array(palm_predictions),
            ood_count, ood_percentage, np.array(distances), instance_info)

def calculate_metrics(targets, predictions):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    
    # Check if it's binary or multi-class
    unique_classes = np.unique(targets)
    if len(unique_classes) == 2:
        auc = roc_auc_score(targets, predictions)
    else:
        # Multi-class scenario
        binarized_targets = label_binarize(targets, classes=unique_classes)
        binarized_predictions = label_binarize(predictions, classes=unique_classes)
        auc = roc_auc_score(binarized_targets, binarized_predictions, average='weighted', multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

def run_test(dataset_name, label_columns, instance_columns, config):
    img_size = config['img_size']
    bag_batch_size = config['bag_batch_size']
    min_bag_size = config['min_bag_size']
    max_bag_size = config['max_bag_size']
    instance_batch_size = config['instance_batch_size']
    arch = config['arch']
    pretrained_arch = config['pretrained_arch']
    num_labels = len(label_columns)

    # Define transforms
    test_transform = T.Compose([
        CLAHETransform(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare PALM
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 90, lambda_pcon=1).cuda()
    palm.load_state(palm_path)

    # Prepare test data
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    bags_train, bags_test = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)

    # Create test datasets and dataloaders
    bag_dataset_test = BagOfImagesDataset(bags_test, transform=test_transform, save_processed=False)
    bag_dataloader_test = TUD.DataLoader(bag_dataset_test, batch_size=bag_batch_size, collate_fn=collate_bag, shuffle=False)

    instance_dataset_test = Instance_Dataset_with_IDs(bags_train, [], transform=test_transform, warmup=True)
    instance_dataloader_test = TUD.DataLoader(instance_dataset_test, batch_size=instance_batch_size, collate_fn=collate_instance, shuffle=False)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(arch, pretrained_arch, num_classes=num_labels).to(device)
    
    # Load the saved model state
    if model_version:
        model_path = f"{model_folder}/{head_name}/{model_version}/model.pth"
    else:
        model_path = f"{model_folder}/{head_name}/model.pth"
    model.load_state_dict(torch.load(model_path))

    # Test the model
    results = test_model_and_collect_distances(model, palm, bag_dataloader_test, instance_dataloader_test, device)
    bag_targets, bag_predictions, instance_targets, fc_predictions, palm_predictions, ood_count, ood_percentage, distances, instance_info = results

    # Calculate and print metrics
    print(f"\nResults for dataset: {dataset_name}")
    print("Bag-level Metrics:")
    bag_metrics = calculate_metrics(bag_targets, bag_predictions)
    for metric, value in bag_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nFC Instance-level Metrics:")
    fc_instance_metrics = calculate_metrics(instance_targets, fc_predictions)
    for metric, value in fc_instance_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nPALM Instance-level Metrics:")
    palm_instance_metrics = calculate_metrics(instance_targets, palm_predictions)
    for metric, value in palm_instance_metrics.items():
        print(f"{metric}: {value:.4f}")

    print(f"\nOut-of-Distribution Instances:")
    print(f"Count: {ood_count}")
    print(f"Percentage: {ood_percentage:.2f}%")

    return distances, instance_info

if __name__ == '__main__':
    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  

    # Load the model configuration
    head_name = "Head_OOD_Testing_efficientnet_b0"
    model_version = "" #Leave "" to read HEA
    
    # loaded configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)
    palm_path = os.path.join(model_folder, head_name, model_version, "palm_state.pkl")

    # Test 1: Original dataset
    distances_1, instance_info_1 = run_test(config['dataset_name'], config['label_columns'], config['instance_columns'], config)
    
    dataset_name = 'export_oneLesions'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present'] 
    
    
    # Test 2: Imagenette dataset
    distances_2, _ = run_test(dataset_name, label_columns, instance_columns, config)
    
    # Create distribution graph
    plt.figure(figsize=(10, 6))
    plt.hist(distances_1, bins=50, alpha=0.5, label=config['dataset_name'])
    plt.hist(distances_2, bins=50, alpha=0.5, label=dataset_name)
    plt.xlabel('Distance to Prototypes')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances to Prototypes')
    plt.legend()
    plt.savefig('prototype_distances_distribution.png')
    plt.show()

    # Identify worst-performing instances (5% with furthest distance) for Test 1
    num_worst = int(0.05 * len(distances_1))
    worst_indices = np.argsort(distances_1)[-num_worst:]
    worst_instances = [instance_info_1[i] for i in worst_indices]

    # Create a directory for the worst-performing instances
    worst_instances_dir = os.path.join(current_dir, "worst_instances")
    os.makedirs(worst_instances_dir, exist_ok=True)

    # Use locate_images to copy the worst-performing instances
    export_location = f'D:/DATA/CASBUSI/exports/{config["dataset_name"]}/'
    locate_images(export_location, worst_instances, worst_instances_dir)

    # Save the worst-performing instances information to a file
    with open(os.path.join(worst_instances_dir, 'worst_performing_instances.txt'), 'w') as f:
        for i, (bag_id, image_index) in enumerate(worst_instances):
            distance = distances_1[worst_indices[i]]
            f.write(f"Bag ID: {bag_id}, Image Index: {image_index}, Distance: {distance}\n")

    print(f"\nWorst-performing instances have been copied to: {worst_instances_dir}")
    print(f"Details saved in: {os.path.join(worst_instances_dir, 'worst_performing_instances.txt')}")
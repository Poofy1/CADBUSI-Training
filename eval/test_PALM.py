import torch, os, sys
import torch.utils.data as TUD
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train_PALM2 import Instance_Dataset, collate_instance
from util.format_data import *
from util.sudo_labels import *
from loss.palm import PALM
from archs.save_arch import *
from archs.model_PALM2_solo import *

def test_model(model, palm, bag_dataloader, instance_dataloader, device):
    model.eval()
    
    # Bag-level metrics
    bag_targets, bag_predictions = [], []
    
    # Instance-level metrics
    instance_targets, fc_predictions, palm_predictions = [], [], []
    
    # Out-of-distribution counter
    ood_count = 0
    total_instances = 0
    
    with torch.no_grad():
        # Bag-level testing
        for images, yb, _, _ in tqdm(bag_dataloader, desc="Testing bags"):
            bag_pred, _, _, _ = model(images, pred_on=True)
            bag_targets.extend(yb.cpu().numpy())
            bag_predictions.extend((bag_pred > 0.5).float().cpu().numpy())
        
        # Instance-level testing
        for images, instance_labels in tqdm(instance_dataloader, desc="Testing instances"):
            images = images.to(device)
            _, _, fc_pred, features = model(images, projector=True)
            palm_pred, dist = palm.predict(features)
            
            # Check for out-of-distribution instances
            ood_instances = (dist > .1).sum().item()
            ood_count += ood_instances
            total_instances += len(instance_labels)
            
            instance_targets.extend(instance_labels.cpu().numpy())
            fc_predictions.extend((fc_pred > 0.5).float().cpu().numpy())
            palm_predictions.extend(palm_pred.cpu().numpy())
    
    ood_percentage = (ood_count / total_instances) * 100
    
    return (np.array(bag_targets), np.array(bag_predictions), 
            np.array(instance_targets), np.array(fc_predictions), np.array(palm_predictions),
            ood_count, ood_percentage)
    
    
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

if __name__ == '__main__':
    
    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  

    # Load the model configuration
    head_name = "Head_Palm2_TEST_324_resnet18"
    model_version = "1"
    
    # loaded configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)
    dataset_name = config['dataset_name']
    label_columns = config['label_columns']
    instance_columns = config['instance_columns']

    #OVERWRITING 
    """dataset_name = 'export_oneLesions' 
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  """
    
    img_size = config['img_size']
    bag_batch_size = config['bag_batch_size']
    min_bag_size = config['min_bag_size']
    max_bag_size = config['max_bag_size']
    instance_batch_size = config['instance_batch_size']
    arch = config['arch']
    pretrained_arch = config['pretrained_arch']
    num_labels = len(label_columns)
    palm_path = os.path.join(model_path, "palm_state.pkl")

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

    instance_dataset_test = Instance_Dataset(bags_test, [], transform=test_transform, warmup=False)
    instance_dataloader_test = TUD.DataLoader(instance_dataset_test, batch_size=instance_batch_size, collate_fn=collate_instance, shuffle=False)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(arch, pretrained_arch, num_classes=num_labels).to(device)
    
    # Load the saved model state
    model_path = f"{model_folder}/{head_name}/{model_version}/model.pth"
    model.load_state_dict(torch.load(model_path))

    # Test the model
    bag_targets, bag_predictions, instance_targets, fc_predictions, palm_predictions, ood_count, ood_percentage = test_model(model, palm, bag_dataloader_test, instance_dataloader_test, device)

    # Calculate and print metrics
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
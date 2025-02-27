import torch
import os
import sys
import torch.utils.data as TUD
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from util.eval_util import *
from data.format_data import *
from data.sudo_labels import *
from loss.palm_proto_alt import PALM
from data.save_arch import *
from archs.model_MIL import *
from data.bag_loader import *
from data.instance_loader import *


def test_model_and_collect_distances(model, palm, bag_dataloader, instance_dataloader, device):
    model.eval()
    
    # Bag-level metrics
    bag_targets, bag_predictions = [], []
    
    # Instance-level metrics
    instance_targets, fc_predictions, palm_predictions = [], [], []
    instance_info = []
    instance_features = []
    distances = []
    
    with torch.no_grad():
        # Bag-level testing
        for images, yb, _, unique_id in tqdm(bag_dataloader, desc="Testing bags"):
            # Handle both padded tensor and list of tensors cases
            if not isinstance(images, list):
                images, yb = images.cuda(), yb.cuda()
            else:
                images = [img.cuda() for img in images]
                yb = yb.cuda()
                
            bag_pred, _, _, _= model(images, pred_on=True)
            bag_targets.extend(yb.cpu().numpy())
            bag_predictions.extend((torch.sigmoid(bag_pred)).float().cpu().numpy())
        
        for images, instance_labels, unique_ids in tqdm(instance_dataloader, desc="Testing instances"):
            images, yb = images.cuda(), yb.cuda()
            _, _, fc_pred, features = model(images, projector=True)
            palm_pred, dist = palm.predict(features)
            
            distances.extend(dist.cpu().numpy())
            instance_info.extend(unique_ids) 
            instance_features.extend(features.cpu().numpy())
            
            instance_targets.extend(instance_labels.cpu().numpy())
            # Check if fc_pred is None and handle accordingly
            if fc_pred is None:
                fc_predictions.extend([0] * len(instance_labels))
            else:
                fc_predictions.extend(torch.sigmoid(fc_pred).float().cpu().numpy())
            palm_predictions.extend(palm_pred.cpu().numpy())
                
    return (np.array(bag_targets), np.array(bag_predictions),  
            np.array(instance_targets), np.array(fc_predictions), np.array(palm_predictions),
            np.array(distances), instance_info, np.array(instance_features))



def run_test(config):
    # Define transforms
    test_transform = T.Compose([
        CLAHETransform(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare PALM
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 0, lambda_pcon=1).cuda()
    palm.load_state(palm_path)

    # Prepare test data
    export_location = f"D:/DATA/CASBUSI/exports/{config['dataset_name']}/"
    cropped_images = f"F:/Temp_SSD_Data/{config['dataset_name']}_{config['img_size']}_images/"
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_test = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

            
    instance_dataset_test = Instance_Dataset(bags_val, [], transform=test_transform, warmup=True)
    instance_dataloader_test = TUD.DataLoader(instance_dataset_test, batch_size=config['instance_batch_size'], collate_fn = collate_instance)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes=num_labels).to(device)
    
    # Load the saved model state
    if model_version:
        save_dir = f"{model_folder}/{config['head_name']}/{model_version}"
    else:
        save_dir = f"{model_folder}/{config['head_name']}"
    model_path = os.path.join(save_dir, "model.pth")
    model.load_state_dict(torch.load(model_path))

    # Test the model
    results = test_model_and_collect_distances(model, palm, bag_dataloader_test, instance_dataloader_test, device)
    bag_targets, bag_predictions, instance_targets, fc_predictions, palm_predictions, distances, instance_info, instance_features = results

    # Extract prototypes
    prototypes = palm.protos.cpu().detach().numpy()

    # Calculate prototype labels based on class counts
    prototype_labels = palm.proto_class_counts.cpu().numpy().argmax(axis=1)
    
    save_dir = os.path.join(save_dir, 'tests')
    os.makedirs(save_dir, exist_ok=True)

    # Visualize prototypes and instances
    visualize_prototypes_and_instances(prototypes, prototype_labels, instance_features, instance_targets, config['dataset_name'], config['head_name'], save_dir)
    
     
    # Calculate and print metrics
    print(f"\nResults for dataset: {config['dataset_name']}")
    print("Bag-level Metrics:")
    calculate_metrics(bag_targets, bag_predictions, save_path=os.path.join(save_dir, 'bag'))

    print("\nFC Instance-level Metrics:")
    calculate_metrics(instance_targets, fc_predictions, instance_info, save_path=os.path.join(save_dir, 'instance'))

    print("\nPALM Instance-level Metrics:")
    calculate_metrics(instance_targets, palm_predictions, instance_info, save_path=os.path.join(save_dir, 'palm'))

    return distances, instance_info


if __name__ == '__main__':
    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  
    os.makedirs(f'{current_dir}/results/PALM_OOD/', exist_ok=True)
    
    # Load the model configuration
    head_name = "TEST305"
    model_version = "1" #Leave "" to read HEAD
    
    # loaded configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)
    config['head_name'] = head_name
    palm_path = os.path.join(model_folder, head_name, model_version, "palm_state.pkl")

    # Test 1: Original dataset
    distances_1, _ = run_test(config)
    
    
    """config['dataset_name'] = 'imagenette_dog'
    config['label_columns'] = ['Has_Highland']
    config['instance_columns'] = ['Has_Highland'] 
    
    
    # Test 2: Imagenette dataset
    distances_2, _ = run_test(config)
    
    
    # Calculate OOD statistics
    threshold, ood_percentage = calculate_ood_stats(distances_1, distances_2)
    
    print(f"\nOOD Statistics:")
    print(f"Threshold (95th percentile of {config['dataset_name']}): {threshold:.4f}")
    print(f"Percentage of OOD samples in {config['dataset_name']}: {ood_percentage:.2f}%")
    
    # Create distribution graph
    plt.figure(figsize=(10, 6))
    plt.hist(distances_1, bins=50, alpha=0.5, label=config['dataset_name'])
    plt.hist(distances_2, bins=50, alpha=0.5, label=config['dataset_name'])
    plt.xlabel('Distance to Prototypes')
    plt.ylabel('Frequency')
    plt.title(f'Distances to Prototypes ({head_name})')
    plt.legend()
    plt.savefig(f'{current_dir}/results/PALM_OOD/{head_name}_prototype_distribution.png')
    plt.show()
    """
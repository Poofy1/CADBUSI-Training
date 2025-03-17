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
from data.bag_loader import *
from data.instance_loader import *
from config import val_transform

def save_bag_results_to_csv(bag_results, output_file):
    """
    Save instance-level predictions to a CSV file, with one row per fc_prediction.
    
    Args:
        bag_results (dict): Dictionary of bag results from test_model_and_collect_distances
        output_file (str): Path to save the CSV file
    """
    import pandas as pd
    import numpy as np
    import torch
    
    rows = []
    
    for bag_id, result in bag_results.items():
        # Convert instance_labels to list format
        if torch.is_tensor(result['instance_labels']):
            instance_labels = result['instance_labels'].cpu().numpy().tolist()
        else:
            instance_labels = list(result['instance_labels'])
        
        # Handle fc_predictions
        fc_predictions = result['fc_predictions']
        if fc_predictions is not None:
            # Ensure fc_predictions is iterable
            if np.isscalar(fc_predictions) or (isinstance(fc_predictions, np.ndarray) and fc_predictions.ndim == 0):
                fc_predictions = [fc_predictions]
            elif isinstance(fc_predictions, np.ndarray):
                fc_predictions = fc_predictions.tolist()
        else:
            fc_predictions = []
            
        # Handle att_predictions
        att_predictions = result['att_predictions']
        if att_predictions is not None:
            # Ensure att_predictions is iterable
            if np.isscalar(att_predictions) or (isinstance(att_predictions, np.ndarray) and att_predictions.ndim == 0):
                att_predictions = [att_predictions]
            elif isinstance(att_predictions, np.ndarray):
                att_predictions = att_predictions.tolist()
        else:
            att_predictions = []
            
        # Ensure all arrays have the same length by padding shorter ones
        max_len = max(len(fc_predictions), len(att_predictions), len(instance_labels))
        fc_predictions = fc_predictions + [None] * (max_len - len(fc_predictions))
        att_predictions = att_predictions + [None] * (max_len - len(att_predictions))
        instance_labels = instance_labels + [None] * (max_len - len(instance_labels))
        
        # Create a row for each fc_prediction
        for i in range(max_len):
            row = {
                'accession_number': bag_id,
                'bag_pred': result['prediction'],
                'bag_target': result['target'],
                'fc_pred': fc_predictions[i] if i < len(fc_predictions) else None,
                'att_pred': att_predictions[i] if i < len(att_predictions) else None,
                'instance_label': instance_labels[i] if i < len(instance_labels) else None,
                'instance_index': i
            }
            rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    
def test_model_and_collect_distances(model, palm, bag_dataloader, instance_dataloader):
    model.eval()
    
    # Initialize separate dictionaries keyed by unique_id
    bag_dict = {}  # Will store bag-level results
    bag_targets, bag_predictions = [], []
    
    # Instance-level metrics
    instance_targets, fc_predictions, palm_predictions = [], [], []
    instance_info = []
    instance_features = []
    distances = []
    
    with torch.no_grad():
        # Bag-level testing
        for images, yb, instance_labels, unique_id in tqdm(bag_dataloader, desc="Testing bags"):
            # Handle both padded tensor and list of tensors cases
            if not isinstance(images, list):
                images, yb = images.cuda(), yb.cuda()
            else:
                images = [img.cuda() for img in images]
                yb = yb.cuda()
                
            bag_pred, att_pred, fc_pred, features = model(images, pred_on=True)
            
            
            bag_targets.extend(yb.cpu().numpy())
            bag_predictions.extend((torch.sigmoid(bag_pred)).float().cpu().numpy())
            
            # Process each bag in the batch
            for i in range(len(unique_id)):
                bag_id = unique_id[i]
                
                # Store all instance-level predictions and apply sigmoid to both att_pred and fc_pred
                bag_dict[bag_id] = {
                    'target': yb[i].cpu().item(),
                    'prediction': torch.sigmoid(bag_pred[i]).float().cpu().item(),
                    'att_predictions': torch.sigmoid(att_pred).float().cpu().numpy() if att_pred is not None else None,  # Apply sigmoid
                    'fc_predictions': torch.sigmoid(fc_pred).float().cpu().numpy() if fc_pred is not None else None,
                    'instance_labels': instance_labels[i]
                }
                
        # Instance-level testing
        for images, instance_labels, unique_ids in tqdm(instance_dataloader, desc="Testing instances"):
            images = images.cuda()
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
            
    instance_results = ( np.array(instance_targets), np.array(fc_predictions), np.array(palm_predictions),
                        np.array(distances), instance_info, np.array(instance_features))
    bag_results = (np.array(bag_targets), np.array(bag_predictions))
    
     
    return bag_dict, instance_results, bag_results



def run_test(config):
    
    # Prepare PALM
    palm = PALM(nviews = 1, num_classes=2, n_protos=100, k = 0, lambda_pcon=1).cuda()
    palm.load_state(palm_path)

    # Prepare test data
    config['bag_batch_size'] = 1 # Makes eval easier
    _, bags_val, _, bag_dataloader_test = prepare_all_data(config)
   
    instance_dataset_test = Instance_Dataset(bags_val, [], transform=val_transform, warmup=True)
    instance_dataloader_test = TUD.DataLoader(instance_dataset_test, batch_size=config['instance_batch_size'], collate_fn = collate_instance)

    # Create Model
    model = build_model(config)    
    
    # Load the saved model state
    if model_version:
        save_dir = f"{model_folder}/{config['head_name']}/{model_version}"
    else:
        save_dir = f"{model_folder}/{config['head_name']}"
    model_path = os.path.join(save_dir, "model.pth")
    model.load_state_dict(torch.load(model_path))

    # Test the model
    bag_dict, instance_results, bag_results = test_model_and_collect_distances(model, palm, bag_dataloader_test, instance_dataloader_test)

    save_dir = os.path.join(save_dir, 'tests')
    os.makedirs(save_dir, exist_ok=True)

    bag_targets, bag_pred = bag_results
    
    # Save all predications
    save_bag_results_to_csv(bag_dict, os.path.join(save_dir, "model_results.csv"))
    
    instance_targets, fc_predictions, palm_predictions, distances, instance_info, instance_features = instance_results
    
    # Extract prototypes
    prototypes = palm.protos.cpu().detach().numpy()
    prototype_labels = palm.proto_class_counts.cpu().numpy().argmax(axis=1) # Calculate prototype labels based on class counts
    visualize_prototypes_and_instances(prototypes, prototype_labels, instance_features, instance_targets, config['dataset_name'], config['head_name'], save_dir)
    
    # Calculate and print metrics
    print(f"\nResults for dataset: {config['dataset_name']}")
    print("Bag-level Metrics:")
    calculate_metrics(bag_targets, bag_pred, save_path=os.path.join(save_dir, 'bag'))

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
    
    # Load the model configuration
    head_name = "TEST315"
    model_version = "" #Leave "" to read HEAD
    
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
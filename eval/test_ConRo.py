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
from data.pseudo_labels import *
from loss.ConRo import *
from data.save_arch import *
from data.bag_loader import *
from data.instance_loader import *
from config import val_transform
    
def visualize_instance_features(instance_features, instance_labels, dataset_name, output_path):
   # Move tensors to CPU and convert to numpy
   if torch.is_tensor(instance_features):
       instance_features = instance_features.detach().cpu().numpy()
   if torch.is_tensor(instance_labels):
       instance_labels = instance_labels.detach().cpu().numpy()

   # Randomly sample 1000 instances if there are more than 1000
   num_instances = instance_features.shape[0]
   if num_instances > 1000:
       sample_indices = np.random.choice(num_instances, 1000, replace=False)
       instance_features = instance_features[sample_indices]
       instance_labels = instance_labels[sample_indices]

   # Apply t-SNE to the features
   tsne = TSNE(n_components=3, random_state=42)
   instances_tsne = tsne.fit_transform(instance_features)

   # Create 3D scatter plot
   fig = go.Figure()

   # Add instances with color based on label
   for label, color in [(0, 'blue'), (1, 'red'), (-1, 'black')]:
       mask = instance_labels == label
       fig.add_trace(go.Scatter3d(
           x=instances_tsne[mask, 0],
           y=instances_tsne[mask, 1],
           z=instances_tsne[mask, 2],
           mode='markers',
           marker=dict(
               size=3,
               color=color,
               opacity=0.5
           ),
           name=f'Instances (Label {label})'
       ))

   # Update layout
   fig.update_layout(
       title=f'Feature Visualization - {dataset_name}',
       scene=dict(
           xaxis_title='t-SNE 1',
           yaxis_title='t-SNE 2',
           zaxis_title='t-SNE 3'
       ),
       width=800,
       height=800,
       margin=dict(r=20, b=10, l=10, t=40)
   )

   # Save the plot as an interactive HTML file
   pio.write_html(fig, file=f'{output_path}/{dataset_name}_TSNE.html')
   print(f"Feature visualization for {dataset_name} saved")
   
   
   
def test_model_and_collect_distances(model, bag_dataloader, instance_dataloader):
    model.eval()
    
    # Initialize separate dictionaries keyed by unique_id
    bag_dict = {}  # Will store bag-level results
    bag_targets, bag_predictions = [], []
    
    # Instance-level metrics
    instance_targets, fc_predictions, palm_predictions = [], [], []
    instance_info = []
    instance_features = []
    
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
            
            instance_info.extend(unique_ids) 
            instance_features.extend(features.cpu().numpy())
            
            instance_targets.extend(instance_labels.cpu().numpy())
            # Check if fc_pred is None and handle accordingly
            if fc_pred is None:
                fc_predictions.extend([0] * len(instance_labels))
            else:
                fc_predictions.extend(torch.sigmoid(fc_pred).float().cpu().numpy())
            
    instance_results = ( np.array(instance_targets), np.array(fc_predictions),
                        instance_info, np.array(instance_features))
    bag_results = (np.array(bag_targets), np.array(bag_predictions))
    
     
    return bag_dict, instance_results, bag_results



def run_test(config):

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
    bag_dict, instance_results, bag_results = test_model_and_collect_distances(model, bag_dataloader_test, instance_dataloader_test)

    save_dir = os.path.join(save_dir, 'tests')
    os.makedirs(save_dir, exist_ok=True)
    
    instance_targets, fc_predictions, instance_info, instance_features = instance_results

    visualize_instance_features(instance_features, instance_targets, config['dataset_name'], save_dir)

if __name__ == '__main__':
    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  
    
    # Load the model configuration
    head_name = "TEST331"
    model_version = "" #Leave "" to read HEAD
    
    # loaded configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)
    config['head_name'] = head_name
    palm_path = os.path.join(model_folder, head_name, model_version, "palm_state.pkl")

    # Test 1: Original dataset
    run_test(config)
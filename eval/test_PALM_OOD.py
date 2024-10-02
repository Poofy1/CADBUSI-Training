import torch
import os
import sys
import torch.utils.data as TUD
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from util.format_data import *
from util.sudo_labels import *
from loss.palm import PALM
from archs.save_arch import *
from archs.model_PALM import *
from data.bag_loader import *
from data.instance_loader import *

    
def visualize_prototypes_and_instances(palm, instance_features, instance_labels, dataset_name, head_name):
    # Extract prototypes
    prototypes = palm.protos.cpu().numpy()

    # Calculate prototype labels based on class counts
    prototype_labels = palm.proto_class_counts.cpu().numpy().argmax(axis=1)

    # Randomly sample 1000 instances if there are more than 1000
    num_instances = instance_features.shape[0]
    if num_instances > 1000:
        sample_indices = np.random.choice(num_instances, 1000, replace=False)
        instance_features = instance_features[sample_indices]
        instance_labels = instance_labels[sample_indices]

    # Combine prototypes and instances
    combined_features = np.vstack((prototypes, instance_features))

    # Apply t-SNE to the combined features
    tsne = TSNE(n_components=3, random_state=42)
    combined_tsne = tsne.fit_transform(combined_features)

    # Split the results back into prototypes and instances
    num_prototypes = prototypes.shape[0]
    prototypes_tsne = combined_tsne[:num_prototypes]
    instances_tsne = combined_tsne[num_prototypes:]

    # Create 3D scatter plot
    fig = go.Figure()

    # Add prototypes
    fig.add_trace(go.Scatter3d(
        x=prototypes_tsne[:, 0],
        y=prototypes_tsne[:, 1],
        z=prototypes_tsne[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=prototype_labels,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f"Prototype {i}, Label: {label}" for i, label in enumerate(prototype_labels)],
        hoverinfo='text',
        name='Prototypes'
    ))

    # Add instances with color based on label
    for label, color in [(0, 'blue'), (1, 'red')]:
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
        title=f'Prototypes/Instances {head_name} {dataset_name}',
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
    pio.write_html(fig, file=f'{current_dir}/results/PALM_OOD/{head_name}_{dataset_name}_TSNE.html')
    print("Prototype and instances visualization saved as 'TSNE.html'")
    

    
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
        for images, yb, _, _ in tqdm(bag_dataloader, desc="Testing bags"):
            bag_pred, _, _= model(images, pred_on=True)
            bag_targets.extend(yb.cpu().numpy())
            bag_predictions.extend((bag_pred > 0.5).float().cpu().numpy())
        
        for images, instance_labels, unique_ids in tqdm(instance_dataloader, desc="Testing instances"):
            images = images.to(device)
            _, fc_pred, features = model(images, projector=True)
            palm_pred, dist = palm.predict(features)
            
            distances.extend(dist.cpu().numpy())
            instance_info.extend(unique_ids) 
            instance_features.extend(features.cpu().numpy())
            
            instance_targets.extend(instance_labels.cpu().numpy())
            # Check if fc_pred is None and handle accordingly
            if fc_pred is None:
                fc_predictions.extend([0] * len(instance_labels))
            else:
                fc_predictions.extend((fc_pred > 0.5).float().cpu().numpy())
            palm_predictions.extend(palm_pred.cpu().numpy())
                
    return (np.array(bag_targets), np.array(bag_predictions), 
            np.array(instance_targets), np.array(fc_predictions), np.array(palm_predictions),
            np.array(distances), instance_info, np.array(instance_features))

def calculate_metrics(targets, predictions):
    accuracy = balanced_accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary')
    recall = recall_score(targets, predictions, average='binary')
    f1 = f1_score(targets, predictions, average='binary')
    
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

def run_test(config):
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
    export_location = f"D:/DATA/CASBUSI/exports/{config['dataset_name']}/"
    cropped_images = f"F:/Temp_SSD_Data/{config['dataset_name']}_{config['img_size']}_images/"
    bags_train, bags_test = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create test datasets and dataloaders
    bag_dataset_test = BagOfImagesDataset(bags_test, transform=test_transform, save_processed=False)
    bag_dataloader_test = TUD.DataLoader(bag_dataset_test, batch_size=config['bag_batch_size'], collate_fn=collate_bag, shuffle=False)

    instance_dataset_test = Instance_Dataset(bags_test, [], transform=test_transform, warmup=True)
    instance_dataloader_test = TUD.DataLoader(instance_dataset_test, batch_size=config['instance_batch_size'], collate_fn=collate_instance, shuffle=False)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes=num_labels).to(device)
    
    # Load the saved model state
    if model_version:
        model_path = f"{model_folder}/{config['head_name']}/{model_version}/model.pth"
    else:
        model_path = f"{model_folder}/{config['head_name']}/model.pth"
    model.load_state_dict(torch.load(model_path))

    # Test the model
    results = test_model_and_collect_distances(model, palm, bag_dataloader_test, instance_dataloader_test, device)
    bag_targets, bag_predictions, instance_targets, fc_predictions, palm_predictions, distances, instance_info, instance_features = results

    # Visualize prototypes and instances
    visualize_prototypes_and_instances(palm, instance_features, instance_targets, config['dataset_name'], config['head_name'])
    
     
    # Calculate and print metrics
    print(f"\nResults for dataset: {config['dataset_name']}")
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

    return distances, instance_info


def calculate_ood_stats(distances_1, distances_2):
    # Calculate threshold (95th percentile of distances_1)
    threshold = np.percentile(distances_1, 95)
    
    # Calculate percentage of OOD samples in distances_2
    ood_percentage = (distances_2 > threshold).mean() * 100
    
    return threshold, ood_percentage

if __name__ == '__main__':
    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  
    os.makedirs(f'{current_dir}/results/PALM_OOD/', exist_ok=True)
    
    # Load the model configuration
    head_name = "PALM_ITS2CLR_CADBUSI_1"
    model_version = "" #Leave "" to read HEAD
    
    # loaded configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)
    config['head_name'] = head_name
    palm_path = os.path.join(model_folder, head_name, model_version, "palm_state.pkl")

    # Test 1: Original dataset
    distances_1, _ = run_test(config)
    
    
    config['dataset_name'] = 'export_oneLesions'
    config['label_columns'] = ['Has_Malignant']
    config['instance_columns'] = ['Malignant Lesion Present'] 
    
    
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

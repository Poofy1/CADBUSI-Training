import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)






def visualize_prototypes_and_instances(prototypes, prototype_labels, instance_features, instance_labels, dataset_name, head_name, output_path):

    # Move tensors to CPU and convert to numpy
    if torch.is_tensor(prototypes):
        prototypes = prototypes.detach().cpu().numpy()
    if torch.is_tensor(prototype_labels):
        prototype_labels = prototype_labels.detach().cpu().numpy()
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
    pio.write_html(fig, file=f'{output_path}/{head_name}_{dataset_name}_TSNE.html')
    print("Prototype and instances visualization saved")
    
    
    


def calculate_ood_stats(distances_1, distances_2):
    # Calculate threshold (95th percentile of distances_1)
    threshold = np.percentile(distances_1, 95)
    
    # Calculate percentage of OOD samples in distances_2
    ood_percentage = (distances_2 > threshold).mean() * 100
    
    return threshold, ood_percentage




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
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
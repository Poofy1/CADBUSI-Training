import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score
import seaborn as sns
import pandas as pd

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




def calculate_metrics(targets, predictions, target_specificity=0.80, save_path="./"):
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Fiolter out non 0/1 classes
    binary_indices = np.where((targets == 0) | (targets == 1))[0]
    targets = targets[binary_indices]
    predictions = predictions[binary_indices]
    
    # Basic metrics
    pred_class = (predictions >= 0.5).astype(int)  # Threshold for basic metrics
    accuracy = balanced_accuracy_score(targets, pred_class)
    precision = precision_score(targets, pred_class, average='binary')
    recall = recall_score(targets, pred_class, average='binary') 
    f1 = f1_score(targets, pred_class, average='binary')
    auc = roc_auc_score(targets, predictions)
    
    # Calculate metrics for threshold 0.5
    pred_class = (predictions >= 0.5).astype(int)
    TP = np.sum((targets == 1) & (pred_class == 1))
    TN = np.sum((targets == 0) & (pred_class == 0))
    FP = np.sum((targets == 0) & (pred_class == 1))
    FN = np.sum((targets == 1) & (pred_class == 0))
    
    default_sens = TP / (TP + FN) if (TP + FN) != 0 else 0
    default_spec = TN / (TN + FP) if (TN + FP) != 0 else 0
    default_ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
    default_npv = TN / (TN + FN) if (TN + FN) != 0 else 0
    
    # Print basic metrics
    print("Basic Metrics:")
    print(f"* Accuracy: {accuracy:.2%}")
    print(f"* AUC: {auc:.2%}")
    print(f"* Sensitivity: {default_sens:.2%}")
    print(f"* Specificity: {default_spec:.2%}")
    print(f"* PPV: {default_ppv:.2%}")
    print(f"* NPV: {default_npv:.2%}")
    print(f"* Threshold: 0.50")
    print(f"* Precision: {precision:.2%}")
    print(f"* Recall: {recall:.2%}")
    print(f"* F1 Score: {f1:.2%}")
    
    # Detailed threshold analysis
    thresholds = np.linspace(0, 1, 1000)
    best_threshold = best_accuracy = best_sensitivity = best_specificity = best_ppv = best_npv = 0
    target_threshold = None
    target_metrics = None
    
    # Arrays to store metrics for plotting
    sensitivity_arr = []
    specificity_arr = []
    accuracy_arr = []
    
    for threshold in thresholds:
        pred_class = (predictions >= threshold).astype(int)
        TP = np.sum((targets == 1) & (pred_class == 1))
        TN = np.sum((targets == 0) & (pred_class == 0))
        FP = np.sum((targets == 0) & (pred_class == 1))
        FN = np.sum((targets == 1) & (pred_class == 0))
        
        sens = TP / (TP + FN) if (TP + FN) != 0 else 0
        spec = TN / (TN + FP) if (TN + FP) != 0 else 0
        acc = (TP + TN) / (TP + TN + FP + FN)
        ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
        npv = TN / (TN + FN) if (TN + FN) != 0 else 0
        
        sensitivity_arr.append(sens)
        specificity_arr.append(spec)
        accuracy_arr.append(acc)
        
        if target_threshold is None and spec >= target_specificity:
            target_threshold = threshold
            target_metrics = (acc, sens, spec, ppv, npv)
            
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold
            best_sensitivity = sens
            best_specificity = spec
            best_ppv = ppv
            best_npv = npv
    
    print("\nBest Accuracy Threshold Metrics:")
    print(f"* Accuracy: {best_accuracy:.2%}")
    print(f"* Sensitivity: {best_sensitivity:.2%}")
    print(f"* Specificity: {best_specificity:.2%}")
    print(f"* PPV: {best_ppv:.2%}")
    print(f"* NPV: {best_npv:.2%}")
    print(f"* Threshold: {best_threshold:.2f}")
    
    if target_metrics:
        print(f"\nTarget Specificity ({target_specificity:.0%}) Threshold Metrics:")
        print(f"* Accuracy: {target_metrics[0]:.2%}")
        print(f"* Sensitivity: {target_metrics[1]:.2%}")
        print(f"* Specificity: {target_metrics[2]:.2%}")
        print(f"* PPV: {target_metrics[3]:.2%}")
        print(f"* NPV: {target_metrics[4]:.2%}")
        print(f"* Threshold: {target_threshold:.2f}")
    
    
    # Plot Performance Metrics
    plt.figure(figsize=(8, 8))
    plt.plot(thresholds, sensitivity_arr, label='Sensitivity')
    plt.plot(thresholds, specificity_arr, label='Specificity')
    plt.plot(thresholds, accuracy_arr, label='Accuracy')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label='Best Accuracy Threshold')
    plt.axvline(x=target_threshold, color='g', linestyle='--', label=f'Target Specificity Threshold ({target_specificity:.0%})')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/Performance_metrics_graph.png")
    plt.close()

    # Plot ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(targets, predictions)
    
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')

    # Find points for thresholds
    target_fpr = 1 - target_metrics[2]
    target_tpr = target_metrics[1]
    
    threshold_05_idx = np.argmin(np.abs(thresholds - 0.5))
    fpr_05 = 1 - specificity_arr[threshold_05_idx]
    tpr_05 = sensitivity_arr[threshold_05_idx]

    # Plot threshold points
    plt.plot(target_fpr, target_tpr, 'ro', markersize=10, label=f'Threshold at {target_specificity:.0%} Specificity')
    plt.plot(fpr_05, tpr_05, 'go', markersize=10, label='Threshold at 0.5')

    # Add crosshairs
    plt.axvline(x=target_fpr, color='red', linestyle=':', alpha=0.8)
    plt.axhline(y=target_tpr, color='red', linestyle=':', alpha=0.8)
    plt.axvline(x=fpr_05, color='green', linestyle=':', alpha=0.8)
    plt.axhline(y=tpr_05, color='green', linestyle=':', alpha=0.8)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Add annotations
    plt.annotate(f'Threshold: {target_threshold:.2f}',
                xy=(target_fpr, target_tpr), xycoords='data',
                xytext=(10, -10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.annotate(f'Threshold: 0.5',
                xy=(fpr_05, tpr_05), xycoords='data',
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.savefig(f"{save_path}/AUC_graph.png")
    plt.close()
    
    print(f'Saved performace graphs to {save_path}')
    
    


def get_worse_instances(targets, predictions, output_path='./'):
    """
    Analyze and identify poor performing instances from targets and predictions arrays
    
    Args:
        targets: Array-like of ground truth labels (0 or 1)
        predictions: Array-like of prediction probabilities (0 to 1)
        output_path: Path to save the poor performing instances CSV
    """
    # Convert inputs to numpy arrays if they aren't already
    targets = np.array(targets)
    predictions = np.array(predictions)
    
    # Filter for only 0 and 1 targets
    valid_mask = (targets == 0) | (targets == 1)
    targets = targets[valid_mask]
    predictions = predictions[valid_mask]
    original_indices = np.where(valid_mask)[0]
    
    # 1. Analyze label distribution
    unique_labels, label_counts = np.unique(targets, return_counts=True)
    label_dist = dict(zip(unique_labels, label_counts))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot label distribution
    ax1.bar(label_dist.keys(), label_dist.values())
    ax1.set_title('Label Distribution')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    
    # Plot prediction distributions by label
    for label in unique_labels:
        label_preds = predictions[targets == label]
        sns.kdeplot(data=label_preds, label=f'Label {label}', ax=ax2)
    
    ax2.set_title('Prediction Distribution by Label')
    ax2.set_xlabel('Prediction Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nLabel Distribution:")
    for label, count in label_dist.items():
        print(f"Label {label}: {count}")
    
    print("\nPrediction Statistics by Label:")
    for label in unique_labels:
        label_preds = predictions[targets == label]
        print(f"\nLabel {label}:")
        print(f"Mean: {np.mean(label_preds):.4f}")
        print(f"Std: {np.std(label_preds):.4f}")
        print(f"Min: {np.min(label_preds):.4f}")
        print(f"Max: {np.max(label_preds):.4f}")

    # Find poor performing instances
    poor_performing_mask = (
        ((targets == 0) & (predictions >= 0.9)) |
        ((targets == 1) & (predictions <= 0.1))
    )
    
    poor_performing_indices = original_indices[poor_performing_mask]
    poor_performing_targets = targets[poor_performing_mask]
    poor_performing_predictions = predictions[poor_performing_mask]
    
    # Create DataFrame and save to CSV
    poor_performing_df = pd.DataFrame({
        'index': poor_performing_indices,
        'targets': poor_performing_targets,
        'predictions': poor_performing_predictions
    })
    
    
    poor_performing_df.to_csv(f'{output_path}/worst_instances.csv', index=False)
    print(f"\nPoor performing instances saved to: {f'{output_path}/worst_instances.csv'}")
    print(f"Number of poor performing instances: {len(poor_performing_indices)}")
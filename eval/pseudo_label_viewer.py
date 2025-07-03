import pickle
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch
# Add parent directory to Python path
env = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(env)
sys.path.append(parent_dir)
from data.format_data import *


def load_and_analyze_matched_data(bags_dict, pseudo_dict_path, save_name):
    """
    Load both datasets and create distribution charts grouped by true labels
    """
    try:
        # Load dictionaries
        with open(pseudo_dict_path, 'rb') as file:
            pseudo_dict = pickle.load(file)
            
        print("Both dictionaries loaded successfully!")
        print(f"Bags dict entries: {len(bags_dict)}")
        print(f"Pseudo dict entries: {len(pseudo_dict)}")
        print(pseudo_dict)
        # Storage for pseudo labels grouped by true labels
        unknown_confidences = []  # true label = -1 (unknown)
        negative_confidences = []  # true label = 0
        positive_confidences = []  # true label = 1
        
        # Process each bag similar to Instance_Dataset
        for bag_id, bag_info in bags_dict.items():
            images = bag_info['images'].copy()
            image_labels = bag_info['image_labels'].copy()
            bag_label = bag_info['bag_labels'][0]
            
            # Get pseudo labels for this bag
            if bag_id in pseudo_dict:
                selection_mask_labels, confidence_scores = pseudo_dict[bag_id]
            else:
                continue  # Skip if no pseudo labels
                
            # Process each instance
            for idx, (img, labels) in enumerate(zip(images, image_labels)):
                if idx >= len(confidence_scores):
                    continue  # Skip if no pseudo label for this instance
                    
                confidence = confidence_scores[idx]
                
                # Skip filtered values (like in your original script)
                if confidence in [-1.0, 0.0, 1.0]:
                    continue
                
                # Determine true label (following Instance_Dataset logic)
                if labels[0] is not None:
                    true_label = labels[0]
                elif bag_label == 0:
                    true_label = 0
                else:
                    true_label = -1  # unknown
                
                # Group pseudo labels by true label
                if true_label == -1:
                    unknown_confidences.append(confidence)
                elif true_label == 0:
                    negative_confidences.append(confidence)
                elif true_label == 1:
                    positive_confidences.append(confidence)
        
        # Print statistics
        print(f"\nPseudo labels by true label:")
        print(f"Unknown labels (-1): {len(unknown_confidences)} instances")
        print(f"Negative labels (0): {len(negative_confidences)} instances")  
        print(f"Positive labels (1): {len(positive_confidences)} instances")
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(env, "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create separate histograms
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Pseudo Label Distribution by True Label', fontsize=16)
        
        # Combined histogram
        ax = axes[0, 0]
        if unknown_confidences:
            ax.hist(unknown_confidences, bins=30, alpha=0.5, label=f'Unknown ({len(unknown_confidences)})', color='gray')
        if negative_confidences:
            ax.hist(negative_confidences, bins=30, alpha=0.5, label=f'Negative ({len(negative_confidences)})', color='red')
        if positive_confidences:
            ax.hist(positive_confidences, bins=30, alpha=0.5, label=f'Positive ({len(positive_confidences)})', color='blue')
        ax.set_title('All Labels Combined')
        ax.set_xlabel('Pseudo Label')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Unknown labels
        ax = axes[0, 1]
        if unknown_confidences:
            ax.hist(unknown_confidences, bins=30, alpha=0.7, color='gray', edgecolor='black')
            ax.axvline(np.mean(unknown_confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(unknown_confidences):.4f}')
            ax.set_title(f'Unknown True Labels ({len(unknown_confidences)} instances)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Unknown Labels', transform=ax.transAxes, ha='center')
            ax.set_title('Unknown True Labels (0 instances)')
        ax.set_xlabel('Pseudo Label')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.3)
        
        # Negative labels (0)
        ax = axes[1, 0]
        if negative_confidences:
            ax.hist(negative_confidences, bins=30, alpha=0.7, color='red', edgecolor='black')
            ax.axvline(np.mean(negative_confidences), color='blue', linestyle='--',
                      label=f'Mean: {np.mean(negative_confidences):.4f}')
            ax.set_title(f'Negative True Labels ({len(negative_confidences)} instances)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Negative Labels', transform=ax.transAxes, ha='center')
            ax.set_title('Negative True Labels (0 instances)')
        ax.set_xlabel('Pseudo Label')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.3)
        
        # Positive labels (1)
        ax = axes[1, 1]
        if positive_confidences:
            ax.hist(positive_confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(positive_confidences), color='red', linestyle='--',
                      label=f'Mean: {np.mean(positive_confidences):.4f}')
            ax.set_title(f'Positive True Labels ({len(positive_confidences)} instances)')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Positive Labels', transform=ax.transAxes, ha='center')
            ax.set_title('Positive True Labels (0 instances)')
        ax.set_xlabel('Pseudo Label')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure
        output_dir = os.path.join(results_dir, "pseudo_labels")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{save_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nGraph saved to: {output_path}")
        
        # Optionally still show the plot
        plt.show()
        
        # Print detailed statistics
        if unknown_confidences:
            print(f"\nUnknown labels stats:")
            print(f"  Mean: {np.mean(unknown_confidences):.4f}")
            print(f"  Median: {np.median(unknown_confidences):.4f}")
            print(f"  Std: {np.std(unknown_confidences):.4f}")
            
        if negative_confidences:
            print(f"\nNegative labels stats:")
            print(f"  Mean: {np.mean(negative_confidences):.4f}")
            print(f"  Median: {np.median(negative_confidences):.4f}")
            print(f"  Std: {np.std(negative_confidences):.4f}")
            
        if positive_confidences:
            print(f"\nPositive labels stats:")
            print(f"  Mean: {np.mean(positive_confidences):.4f}")
            print(f"  Median: {np.median(positive_confidences):.4f}")
            print(f"  Std: {np.std(positive_confidences):.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage
if __name__ == "__main__":
    config = build_config()
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    
    
    model_folder = os.path.join(parent_dir, "models")  
    config['head_name'] = head_name
    config['model_version'] = model_version
    pseudo_dict_path = os.path.join(model_folder, head_name, model_version, "selection_mask.pkl")
    save_name = f"{head_name}_{model_version}_psuedo_labels"

    load_and_analyze_matched_data(bags_train, pseudo_dict_path, save_name)
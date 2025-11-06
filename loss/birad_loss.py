import torch
import torch.nn as nn
import torch.nn.functional as F

def birads_multilabel_loss(birads_pred, bag_descriptions):
    """
    Calculate BCE loss for BI-RADS predictions, ignoring -1 (unknown) labels.
    
    Args:
        birads_pred: List of 6 tensors [batch_size, num_classes] from model
        bag_descriptions: List of batch_size lists, each containing 6 sublists
                         e.g., [[[-1,-1,...], [0,1,...], ...], ...]
    
    Returns:
        loss: Average BCE loss across all non-(-1) labels
    """
    total_loss = 0.0
    total_valid_labels = 0
    
    batch_size = len(bag_descriptions)
    
    # Process each category
    for cat_idx, category_preds in enumerate(birads_pred):
        # Extract ground truth for this category across all bags
        category_targets = []
        for bag_idx in range(batch_size):
            category_targets.append(bag_descriptions[bag_idx][cat_idx])
        
        # Convert to tensor [batch_size, num_classes]
        category_targets = torch.tensor(category_targets, dtype=torch.float32, device=category_preds.device)
        
        # Create mask for valid labels (not -1)
        valid_mask = (category_targets != -1)
        
        # Only compute loss where we have valid labels
        if valid_mask.sum() > 0:
            valid_preds = category_preds[valid_mask]
            valid_targets = category_targets[valid_mask]
            
            # Binary cross entropy loss
            loss = F.binary_cross_entropy(valid_preds, valid_targets, reduction='sum')
            
            total_loss += loss
            total_valid_labels += valid_mask.sum().item()
    
    # Average over all valid labels
    if total_valid_labels > 0:
        return total_loss / total_valid_labels
    else:
        return torch.tensor(0.0, device=birads_pred[0].device, requires_grad=True)
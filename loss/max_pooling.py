import torch
import torch.nn.functional as F

def mil_max_loss(instance_preds, bag_labels, split_sizes):
    """
    instance_preds: [N, 1] raw logits or probabilities
    bag_labels: [B] bag labels (0 or 1)
    """
    instance_probs = torch.sigmoid(instance_preds)

    # Split instances by bag
    bag_preds = []
    start_idx = 0

    for bag_idx, bag_size in enumerate(split_sizes):
        bag_instances = instance_probs[start_idx:start_idx + bag_size]
        bag_pred = torch.max(bag_instances)
        bag_preds.append(bag_pred)
        start_idx += bag_size

    bag_preds = torch.stack(bag_preds)
    
    # Handle shape mismatch for single bag case
    bag_labels_flat = bag_labels.squeeze()
    
    # Ensure both tensors have same number of dimensions
    if bag_labels_flat.dim() == 0:  # scalar
        bag_labels_flat = bag_labels_flat.unsqueeze(0)  # Make it [1]
    
    # FIX: Ensure both tensors have the same dtype
    bag_labels_flat = bag_labels_flat.to(bag_preds.dtype)
    
    return F.binary_cross_entropy(bag_preds, bag_labels_flat)
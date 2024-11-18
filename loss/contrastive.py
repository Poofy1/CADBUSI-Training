import torch
import torch.nn.functional as F
from torch import nn


def contrastive_loss(features, labels, temperature=0.07):
    """
    Compute contrastive loss (InfoNCE/NT-Xent) for the features
    Args:
        features: tensor of shape [batch_size, feature_dim]
        labels: tensor of shape [batch_size]
        temperature: scalar temperature parameter
    """
    # Normalize features
    #features = F.normalize(features, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Create masks for positive and negative pairs
    labels = labels.view(-1, 1)
    mask_positive = (labels == labels.T).float()
    mask_negative = (labels != labels.T).float()
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # Compute log_prob
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    
    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask_positive * log_prob).sum(1) / mask_positive.sum(1)
    
    # Loss
    loss = -mean_log_prob_pos.mean()
    
    return loss
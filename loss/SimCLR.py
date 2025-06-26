import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features_orig, features_aug):
        """
        Compute SimCLR loss for a batch of features.
        
        Args:
            features_orig: [batch_size, feature_dim] - original image features
            features_aug: [batch_size, feature_dim] - augmented image features
        
        Returns:
            loss: SimCLR contrastive loss
        """
        batch_size = features_orig.shape[0]
        assert features_orig.shape == features_aug.shape, "Original and augmented features must have same shape"
        
        # Concatenate original and augmented features
        features = torch.cat([features_orig, features_aug], dim=0)  # [2*batch_size, feature_dim]
        
        # Normalize features to unit vectors (you should uncomment this!)
        #features = F.normalize(features, dim=1)
        
        # Compute similarity matrix: [2*batch_size, 2*batch_size]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels for positive pairs
        # For sample i, its positive is at position i + batch_size (and vice versa)
        labels = torch.cat([torch.arange(batch_size, 2*batch_size), 
                           torch.arange(0, batch_size)]).to(features.device)
        
        # Mask to remove self-similarity (diagonal elements)
        mask = torch.eye(2*batch_size, dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
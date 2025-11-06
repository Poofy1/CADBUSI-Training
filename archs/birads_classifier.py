import torch
import torch.nn as nn
import torch.nn.functional as F

class BIRADSDescriptionClassifier(nn.Module):
    """
    Multi-label classifier for BI-RADS ultrasound descriptions.
    
    6 separate FC heads for each BI-RADS category:
    - Margin (6 classes)
    - Shape (3 classes)
    - Orientation (2 classes)
    - Echo pattern (6 classes)
    - Posterior features (4 classes)
    - Lesion boundary (3 classes)
    """
    def __init__(self, nf):
        super(BIRADSDescriptionClassifier, self).__init__()
        
        # Create separate FC head for each category
        self.margin_head = nn.Linear(nf, 6)
        self.shape_head = nn.Linear(nf, 3)
        self.orientation_head = nn.Linear(nf, 2)
        self.echo_pattern_head = nn.Linear(nf, 6)
        self.posterior_features_head = nn.Linear(nf, 4)
        self.lesion_boundary_head = nn.Linear(nf, 3)
        
        self.heads = [
            self.margin_head,
            self.shape_head,
            self.orientation_head,
            self.echo_pattern_head,
            self.posterior_features_head,
            self.lesion_boundary_head
        ]
        
        self.total_birads_features = 24  # 6+3+2+6+4+3
    
    def forward(self, features, split_sizes):
        """
        Args:
            features: [total_images, nf] feature tensor (all images concatenated)
            split_sizes: List of integers indicating number of images per bag
            
        Returns:
            enriched_features: [total_images, nf + 24] features with BI-RADS concatenated
            bag_level_preds: List of 6 tensors, one for each category
                            Each tensor has shape [num_bags, num_classes_for_category]
        """
        # Get image-level predictions for all categories
        image_level_preds = []
        for head in self.heads:
            logits = head(features)
            probs = torch.sigmoid(logits)  # Multi-label: each can be 0 or 1
            image_level_preds.append(probs)
        
        # Aggregate to bag-level using max pooling
        bag_level_preds = []
        for category_preds in image_level_preds:
            # Split by bags
            category_per_bag = torch.split(category_preds, split_sizes, dim=0)
            # Max pool over each bag
            bag_preds = torch.stack([bag.max(dim=0)[0] for bag in category_per_bag])
            bag_level_preds.append(bag_preds)
        
        # Concatenate all bag-level predictions into one tensor
        birads_features = torch.cat(bag_level_preds, dim=1)  # [num_bags, 24]
        
        # Expand bag-level features back to image-level
        expanded_birads = []
        for bag_idx, bag_size in enumerate(split_sizes):
            # Repeat this bag's BI-RADS features for each image in the bag
            bag_birads = birads_features[bag_idx].unsqueeze(0).expand(bag_size, -1)
            expanded_birads.append(bag_birads)
        
        expanded_birads = torch.cat(expanded_birads, dim=0)  # [total_images, 24]
        
        # Concatenate BI-RADS features to original features
        enriched_features = torch.cat([features, expanded_birads], dim=1)  # [total_images, nf + 24]
        
        return enriched_features, bag_level_preds
    
    def reset_parameters(self):
        """Reset parameters of all heads"""
        for head in self.heads:
            head.reset_parameters()
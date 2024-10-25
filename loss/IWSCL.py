from torch import nn
import torch
import torch.nn.functional as F

class IWSCL(nn.Module):
    def __init__(self, feat_dim, num_classes=2, momentum=0.999, temperature=0.07):
        super(IWSCL, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.proto_class_counts = torch.zeros(num_classes, num_classes).cuda()
        self.momentum = momentum
        self.num_classes = num_classes
        self.temperature = temperature

    def predict(self, features):
        # Assign the majority class to each prototype based on class counts
        _, proto_classes = torch.max(self.proto_class_counts, dim=1)
        
        # Compute the similarity between input features and prototypes
        similarity = torch.matmul(features, self.prototypes.T)
        #print("Similarities:", similarity[:5])
        
        # Get the index of the prototype with the highest similarity
        _, prototype_indices = torch.max(similarity, dim=1)
        
        # Map the prototype indices to their corresponding class labels
        predicted_classes = proto_classes[prototype_indices]
        
        return predicted_classes
    
    def forward(self, features, instance_predictions, instance_labels, queue, queue_labels, val_on=False):
        """
        Args:
            features: Current batch instance features [B, feat_dim]
            instance_predictions: Predicted labels from classifier [B, num_classes]
            instance_labels: Ground truths, -1 if unknown
            queue: Feature queue [Q, feat_dim]
            queue_labels: Labels for queue features [Q, num_classes]
            val_on: Whether in validation mode
        """
 
        # Fix queue shape
        queue = queue.T
        queue = F.normalize(queue.T, dim=1)
        
        """# Compute pairwise distances
        distances = torch.cdist(features, features, p=2)  # Euclidean distance matrix
        
        # Create masks for positive and negative pairs
        labels_matrix = instance_labels.expand(len(instance_labels), len(instance_labels))
        positive_mask = labels_matrix.eq(labels_matrix.T)
        negative_mask = ~positive_mask
        
        # Remove diagonal elements (self-similarity)
        mask_no_diagonal = ~torch.eye(len(instance_labels), dtype=torch.bool, device=features.device)
        positive_mask = positive_mask & mask_no_diagonal
        negative_mask = negative_mask & mask_no_diagonal
        
        # Positive loss: pull same-class samples together
        positive_loss = (distances * positive_mask).sum() / positive_mask.sum()
        
        # Negative loss: push different-class samples apart
        # Use max(0, margin - distance) to create a margin between classes
        margin = 2.0
        negative_loss = torch.clamp(margin - distances, min=0.0)
        negative_loss = (negative_loss * negative_mask).sum() / negative_mask.sum()
        
        # Total loss
        loss = positive_loss + negative_loss * 2"""
        

        instance_predictions = instance_predictions.squeeze()
        
        ground_truth_mask = instance_labels != -1
        pred_labels = (instance_predictions > 0.5).long()
        
        # Override pred labels with ground truth when available
        pred_labels[ground_truth_mask] = instance_labels[ground_truth_mask]

        # Calculate loss for each feature in the batch
        losses = []
        for i in range(len(features)):
            # Current feature and its predicted label
            q_i = features[i:i+1]  # [1, feat_dim]
            y_i = pred_labels[i]
            
            # Create family set (same predicted class)
            family_mask = (queue_labels == y_i)  # [Q]
            family_samples = queue[:, family_mask]  # [feat_dim, num_family]
            
            # Create non-family set (different predicted class)
            nonfamily_mask = (queue_labels != y_i)  # [Q]
            nonfamily_samples = queue[:, nonfamily_mask]  # [feat_dim, num_nonfamily]
            
            if family_mask.sum() == 0 or nonfamily_mask.sum() == 0:
                continue
                
            # Calculate similarities
            l_pos = torch.mm(q_i, family_samples) / self.temperature  # [1, num_family]
            l_neg = torch.mm(q_i, nonfamily_samples) / self.temperature  # [1, num_nonfamily]
                    
            """# Calculate loss for this instance
            logits = torch.cat([l_pos, l_neg], dim=1)  # [1, num_family + num_nonfamily]
            labels = torch.zeros(1, device=features.device, dtype=torch.long)  # positive is first
            instance_loss = F.cross_entropy(logits, labels)"""
            
            #Use InfoNCE-style loss with better numerical stability
            exp_pos = torch.exp(l_pos)
            exp_neg = torch.exp(l_neg)
            instance_loss = -torch.log(
                exp_pos.sum() / (exp_pos.sum() + exp_neg.sum() + 1e-6)
            )

            losses.append(instance_loss)
            
        if len(losses) == 0:
            loss = torch.tensor(0.0, device=features.device)
        else:
            loss = torch.stack(losses).mean()



        # Update prototypes
        if not val_on:
            with torch.no_grad():
                for c in range(self.num_classes):
                    # First use known labels
                    known_mask = instance_labels == c
                    # Then add predictions where labels are unknown (-1)
                    unknown_mask = (instance_labels == -1) & (instance_predictions == c)
                    # Combine masks
                    class_mask = known_mask | unknown_mask
                    class_features = features[class_mask]
                    if class_features.size(0) > 0:
                        new_prototype = class_features.mean(dim=0)
                        self.prototypes.data[c] = self.momentum * self.prototypes.data[c] + (1 - self.momentum) * new_prototype
                        self.prototypes.data[c] = F.normalize(self.prototypes.data[c], dim=0)
                        
                        # Update class counts for this prototype
                        true_labels = instance_labels[class_mask]
                        valid_labels = true_labels != -1
                        if valid_labels.any():
                            for label in true_labels[valid_labels]:
                                self.proto_class_counts[c, label] += 1
        
        
        cosine_sim = F.cosine_similarity(
            self.prototypes[0:1], 
            self.prototypes[1:2]
        )
        #print("Cosine similarity between prototypes:", cosine_sim.item())
        
        
        # Generate pseudo labels using the predict method
        pseudo_labels = self.predict(features)
        #print(pseudo_labels)
        
        # Override pseudo labels with ground truth when available
        pseudo_labels[ground_truth_mask] = instance_labels[ground_truth_mask]

        return loss, pseudo_labels
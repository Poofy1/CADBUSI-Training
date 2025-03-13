import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Implementation of the supervised contrastive loss from the ConRo framework.
    This loss pulls together samples of the same class and pushes apart samples of different classes.
    """
    def __init__(self, temperature=1.0):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels, auxiliary_embeddings=None, auxiliary_labels=None):
        """
        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            labels: Tensor of shape [batch_size]
            auxiliary_embeddings: Optional tensor of shape [auxiliary_batch_size, embedding_dim]
            auxiliary_labels: Optional tensor of shape [auxiliary_batch_size]
        """
        device = embeddings.device
        
        # Combine main batch with auxiliary batch if provided
        if auxiliary_embeddings is not None and auxiliary_labels is not None:
            all_embeddings = torch.cat([embeddings, auxiliary_embeddings], dim=0)
            all_labels = torch.cat([labels, auxiliary_labels], dim=0)
        else:
            all_embeddings = embeddings
            all_labels = labels
            
        # embeddings already normalized
        
        # Compute cosine similarity
        cos_sim = torch.matmul(all_embeddings, all_embeddings.T) / self.temperature
        
        # For each sample, exclude comparing with itself
        mask_no_self = ~torch.eye(all_embeddings.shape[0], dtype=torch.bool, device=device)
        
        # For each normal session in the main batch
        normal_mask = labels == 0
        normal_indices = torch.where(normal_mask)[0]
        
        loss = 0.0
        
        # Process only for normal sessions in the main batch (Eq. 1 in the paper)
        for i in normal_indices:
            # A(xi) - all sessions except the current one
            a_xi_indices = torch.where(mask_no_self[i])[0]
            
            # B0(xi) - normal sessions in A(xi)
            b0_xi_indices = torch.where(mask_no_self[i] & (all_labels == 0))[0]
            
            if len(b0_xi_indices) > 0:
                # Sum over all positive pairs
                pair_losses = 0.0
                
                for p in b0_xi_indices:
                    # Calculate l(zi, zp, A(xi)) as in Eq. 2
                    numerator = torch.exp(cos_sim[i, p])
                    denominator = torch.sum(torch.exp(cos_sim[i, a_xi_indices]))
                    
                    if denominator > 0:
                        pair_loss = -torch.log(numerator / denominator)
                        pair_losses += pair_loss
                
                # Average over all positive pairs in B0(xi)
                if len(b0_xi_indices) > 0:
                    loss += pair_losses / len(b0_xi_indices)
        
        # Normalize by the number of normal samples (Eq. 1)
        if len(normal_indices) > 0:
            loss /= len(normal_indices)
            
        return loss

class DeepSVDDLoss(nn.Module):
    """
    Implementation of the DeepSVDD loss from the ConRo framework.
    This loss pushes normal samples inside a minimum volume hypersphere.
    """
    def __init__(self):
        super(DeepSVDDLoss, self).__init__()
        
    def forward(self, embeddings, labels):
        # Calculate the center of normal sessions
        normal_mask = labels == 0
        if not torch.any(normal_mask):
            return torch.tensor(0.0, device=embeddings.device)
            
        v_0 = embeddings[normal_mask].mean(dim=0)
        
        # Calculate squared distances from each sample to center
        # Only include normal sessions in the sum (1-y_i)
        distances = torch.sum((embeddings - v_0) ** 2, dim=1)
        loss = ((1 - labels.float()) * distances).sum() / embeddings.shape[0]
        
        return loss


class ConRoLoss(nn.Module):
    """
    Combined loss for the ConRo framework.
    """
    def __init__(self, temperature=1.0, alternating=True):
        super(ConRoLoss, self).__init__()
        self.sup_contrastive_loss = SupervisedContrastiveLoss(temperature)
        self.deep_svdd_loss = DeepSVDDLoss()
        self.alternating = alternating
        self.current_loss = "contrastive" #contrastive"  # Start with contrastive loss
        
    def forward(self, embeddings, labels, auxiliary_embeddings=None, auxiliary_labels=None):
        """
        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            labels: Tensor of shape [batch_size]
            auxiliary_embeddings: Optional tensor of shape [auxiliary_batch_size, embedding_dim]
            auxiliary_labels: Optional tensor of shape [auxiliary_batch_size]
        """
        if self.alternating:
            # Alternating optimization approach as described in the paper
            if self.current_loss == "contrastive":
                loss = self.sup_contrastive_loss(embeddings, labels, auxiliary_embeddings, auxiliary_labels)
                self.current_loss = "svdd"
            else:
                loss = self.deep_svdd_loss(embeddings, labels)
                self.current_loss = "contrastive"
        else:
            # Joint optimization approach (used in ablation study)
            contrastive_loss = self.sup_contrastive_loss(embeddings, labels, auxiliary_embeddings, auxiliary_labels)
            svdd_loss = self.deep_svdd_loss(embeddings, labels)
            loss = contrastive_loss + svdd_loss
            
        return loss 
    
    
class ConRoStage2Loss(nn.Module):
    """
    Implementation of the Stage 2 loss for the ConRo framework (Equation 6 in the paper).
    """
    def __init__(self, temperature=1.0):
        super(ConRoStage2Loss, self).__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels, auxiliary_embeddings, auxiliary_labels, 
                similar_malicious_embeddings, diverse_malicious_embeddings):
        """
        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            labels: Tensor of shape [batch_size]
            auxiliary_embeddings: Tensor of shape [auxiliary_batch_size, embedding_dim]
            auxiliary_labels: Tensor of shape [auxiliary_batch_size]
            similar_malicious_embeddings: Dict mapping indices to similar potential malicious embeddings
            diverse_malicious_embeddings: Dict mapping indices to diverse potential malicious embeddings
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        
        # Find malicious sessions in the main batch
        malicious_mask = labels == 1
        malicious_indices = torch.where(malicious_mask)[0]
        
        if len(malicious_indices) == 0:
            return torch.tensor(0.0, device=device)
        
        # Combine main and auxiliary batches to form the initial set
        all_embeddings = torch.cat([embeddings, auxiliary_embeddings], dim=0)
        all_labels = torch.cat([labels, auxiliary_labels], dim=0)
        
        # Normalize embeddings
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        
        loss = 0.0
        
        for i in malicious_indices:
            # Get the original malicious embedding
            zi = embeddings[i]
            
            # Get similar and diverse malicious embeddings for this sample
            similar_embs = similar_malicious_embeddings.get(i.item(), torch.tensor([], device=device))
            diverse_embs = diverse_malicious_embeddings.get(i.item(), torch.tensor([], device=device))
            
            if len(similar_embs) == 0 or len(diverse_embs) == 0:
                continue
                
            # A(xi) - all sessions except the current one
            a_xi_indices = torch.arange(len(all_embeddings), device=device)
            a_xi_indices = a_xi_indices[a_xi_indices != i]
            a_xi = all_embeddings[a_xi_indices]
            a_xi_labels = all_labels[a_xi_indices]
            
            # B1(xi) - malicious sessions in A(xi)
            b1_xi_indices = torch.where(a_xi_labels == 1)[0]
            b1_xi = a_xi[b1_xi_indices]
            
            # Normalize generated embeddings
            similar_embs = F.normalize(similar_embs, p=2, dim=1)
            diverse_embs = F.normalize(diverse_embs, p=2, dim=1)
            
            # Construct C(xi) = A(xi) ∪ Ĝb1(xi) ∪ Ĝe1(xi)
            c_xi = torch.cat([a_xi, similar_embs, diverse_embs], dim=0)
            
            # Construct D(xi) = B1(xi) ∪ Ĝb1(xi) ∪ Ĝe1(xi) 
            d_xi = torch.cat([b1_xi, similar_embs, diverse_embs], dim=0)
            
            # Calculate individual loss for each sample in D(xi)
            pair_losses = 0.0
            for zp in d_xi:
                # Calculate cosine similarity between zi and all embeddings in C(xi)
                cos_sim_zi_cxi = torch.matmul(zi.unsqueeze(0), c_xi.T).squeeze(0) / self.temperature
                
                # Calculate cosine similarity between zi and zp
                cos_sim_zi_zp = torch.matmul(zi.unsqueeze(0), zp.unsqueeze(0).T).squeeze() / self.temperature
                
                # Calculate l(zi, zp, C(xi)) as in Equation 7
                numerator = torch.exp(cos_sim_zi_zp)
                denominator = torch.sum(torch.exp(cos_sim_zi_cxi))
                
                if denominator > 0:
                    pair_loss = -torch.log(numerator / denominator)
                    pair_losses += pair_loss
            
            # Average over all samples in D(xi)
            if len(d_xi) > 0:
                loss += pair_losses / len(d_xi)
        
        # Normalize by the batch size (1/R in Equation 6)
        loss /= batch_size
            
        return loss
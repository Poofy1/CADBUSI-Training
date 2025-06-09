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
       
   def forward(self, embeddings, labels, similar_malicious_embeddings, diverse_malicious_embeddings, 
               auxiliary_embeddings=None, auxiliary_labels=None):
       """
       Args:
           embeddings: Tensor of shape [batch_size, embedding_dim]
           labels: Tensor of shape [batch_size]
           similar_malicious_embeddings: Dict mapping indices to similar potential malicious embeddings
           diverse_malicious_embeddings: Dict mapping indices to diverse potential malicious embeddings
           auxiliary_embeddings: Optional tensor of shape [auxiliary_batch_size, embedding_dim]
           auxiliary_labels: Optional tensor of shape [auxiliary_batch_size]
       """
       device = embeddings.device
       batch_size = embeddings.shape[0]
       
       # Find malicious sessions in the main batch
       malicious_mask = labels == 1
       malicious_indices = torch.where(malicious_mask)[0]
       
       if len(malicious_indices) == 0:
           return torch.tensor(0.0, device=device)
       
       # Combine main and auxiliary batches to form the initial set if provided
       if auxiliary_embeddings is not None and auxiliary_labels is not None:
           all_embeddings = torch.cat([embeddings, auxiliary_embeddings], dim=0)
           all_labels = torch.cat([labels, auxiliary_labels], dim=0)
       else:
           all_embeddings = embeddings
           all_labels = labels
       
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



from tqdm import tqdm
def compute_global_v0(dataloader, model, device='cuda'):
    """
    Compute the center of normal samples (v0) using all training data.
    Also reports statistics on how many samples fall within the normal hypersphere.
    
    Args:
        dataloader: DataLoader containing training data
        model: The feature extraction model
        device: Device to run computations on
        
    Returns:
        v0: Tensor representing the center of normal samples
        radius: Tensor representing the radius of normal samples
    """
    model.eval()
    normal_features = []
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Computing v0"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Extract features using the model
            _, _, features = model(images, projector=True)
            
            # Store all features and labels for later analysis
            all_features.append(features)
            all_labels.append(labels)
            
            # Filter normal samples
            normal_mask = labels == 0
            if torch.any(normal_mask):
                normal_features.append(features[normal_mask])
    
    # Combine all features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute v0 from normal features
    if normal_features:
        all_normal_features = torch.cat(normal_features, dim=0)
        v0 = all_normal_features.mean(dim=0)
        
        # Calculate radius
        distances = torch.sum((all_normal_features - v0) ** 2, dim=1)
        radius = distances.mean().sqrt()
    else:
        # Fallback if no normal samples found
        feature_dim = all_features.shape[1]
        v0 = torch.zeros(feature_dim, device=device)
        radius = torch.tensor(1.0, device=device)
    
    # Count samples inside the radius
    all_distances = torch.sum((all_features - v0) ** 2, dim=1)
    inside_mask = all_distances <= radius**2
    
    # Count statistics
    total_samples = len(all_labels)
    total_normal = (all_labels == 0).sum().item()
    total_malicious = (all_labels == 1).sum().item()
    
    normal_inside = (inside_mask & (all_labels == 0)).sum().item()
    normal_outside = total_normal - normal_inside
    
    malicious_inside = (inside_mask & (all_labels == 1)).sum().item()
    malicious_outside = total_malicious - malicious_inside
    
    # Print statistics
    print(f"\nHypersphere Statistics (radius = {radius.item():.4f}):")
    print(f"Total samples: {total_samples}")
    print(f"Normal samples: {total_normal} total")
    print(f"  - Inside hypersphere: {normal_inside} ({normal_inside/total_normal*100:.2f}%)")
    print(f"  - Outside hypersphere: {normal_outside} ({normal_outside/total_normal*100:.2f}%)")
    print(f"Malicious samples: {total_malicious} total")
    print(f"  - Inside hypersphere: {malicious_inside} ({malicious_inside/total_malicious*100:.2f}% of malicious)")
    print(f"  - Outside hypersphere: {malicious_outside} ({malicious_outside/total_malicious*100:.2f}% of malicious)")
    print(f"Contamination within sphere: {malicious_inside/(normal_inside+malicious_inside)*100:.2f}% of samples inside are malicious")
    
    return v0, radius
    
def generate_stage2_features(features, labels, beta1=0.92, beta2=4.0, v0=None):
   """
   Generate similar and diverse potential malicious embeddings for ConRo Stage 2
   
   Args:
       features: Tensor of shape [batch_size, embedding_dim] - normalized feature embeddings
       labels: Tensor of shape [batch_size] - binary labels (0=normal, 1=malicious)
       beta1: Float parameter controlling similarity of generated similar embeddings (default: 0.92)
       beta2: Float parameter controlling diversity of generated diverse embeddings (default: 4.0)
       v0: Optional tensor representing center of normal samples (if None, will be calculated)
       
   Returns:
       similar_malicious_embeddings: Dict mapping indices to similar potential malicious embeddings
       diverse_malicious_embeddings: Dict mapping indices to diverse potential malicious embeddings
   """
   device = features.device
   
   # Find malicious indices
   malicious_indices = torch.where(labels == 1)[0]
   
   # If no malicious samples, return empty dictionaries
   if len(malicious_indices) == 0:
       return {}, {}
   
   # Calculate v0 (center of normal samples) if not provided
   if v0 is None:
       normal_mask = labels == 0
       if torch.any(normal_mask):
           v0 = features[normal_mask].mean(dim=0)
       else:
           # If no normal samples, use origin as fallback
           v0 = torch.zeros(features.shape[1], device=device)
   
   # Calculate radius of normal hypersphere (optional)
   normal_mask = labels == 0
   if torch.any(normal_mask):
       distances = torch.sum((features[normal_mask] - v0) ** 2, dim=1)
       radius = distances.mean().sqrt()  # Can adjust this for strictness
   else:
       radius = 1.0  # Default fallback
   
   similar_malicious_embeddings = {}
   diverse_malicious_embeddings = {}
   
   # For each malicious sample
   for idx in malicious_indices:
       # Get original embedding
       zi = features[idx]
       
       # Get other malicious samples for mixing
       other_malicious = features[labels == 1]
       
       # Skip if we only have one malicious sample
       if len(other_malicious) <= 1:
           continue
           
       # Generate similar malicious embeddings (Gb1)
       num_similar = 20  # Number of similar embeddings to generate
       similar_embeds = []
       
       for _ in range(num_similar):
           # Sample random malicious embedding (excluding current one)
           other_idx = torch.randint(0, len(other_malicious), (1,)).item()
           if other_malicious[other_idx].equal(zi):
               continue
               
           zj = other_malicious[other_idx]
           
           # Generate lambda1 from U(beta1, 1)
           lambda1 = torch.empty(1, device=device).uniform_(beta1, 1.0)
           
           # Mix embeddings
           mixed = lambda1 * zi + (1 - lambda1) * zj
           
           # Normalize
           mixed = F.normalize(mixed, p=2, dim=0)
           
           similar_embeds.append(mixed)
           
       if similar_embeds:
           similar_malicious_embeddings[idx.item()] = torch.stack(similar_embeds)
       
       # Generate diverse malicious embeddings (Ge1)
       num_diverse = 200  # Number of diverse embeddings to generate
       diverse_embeds = []
       
       attempts = 0
       max_attempts = num_diverse * 3  # Allow more attempts since many might be filtered
       
       while len(diverse_embeds) < num_diverse and attempts < max_attempts:
           attempts += 1
           
           # Sample random malicious embedding
           other_idx = torch.randint(0, len(other_malicious), (1,)).item()
           zj = other_malicious[other_idx]
           
           # Generate lambda2 from U(-beta2, beta2)
           lambda2 = torch.empty(1, device=device).uniform_(-beta2, beta2)
           
           # Mix embeddings
           mixed = lambda2 * zi + (1 - lambda2) * zj
           
           # Normalize
           mixed = F.normalize(mixed, p=2, dim=0)
           
           # Filter function - only keep if outside normal hypersphere
           distance_to_v0 = torch.sum((mixed - v0) ** 2)
           if distance_to_v0 > radius**2:  # Outside normal hypersphere
               diverse_embeds.append(mixed)
               
       if diverse_embeds:
           diverse_malicious_embeddings[idx.item()] = torch.stack(diverse_embeds)
   
   return similar_malicious_embeddings, diverse_malicious_embeddings
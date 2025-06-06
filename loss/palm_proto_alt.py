import torch
from data.format_data import *



class PALM(nn.Module):
    def __init__(self, nviews, num_classes=2, n_protos=50, proto_m=0.99, temp=0.1, lambda_pcon=1, k=5, feat_dim=128, epsilon=0.05):
        super(PALM, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  # temperature scaling
        self.nviews = nviews
        self.cache_size = int(n_protos / num_classes)
        
        self.lambda_pcon = lambda_pcon
        
        self.feat_dim = feat_dim
        
        self.epsilon = epsilon
        self.sinkhorn_iterations = 3
        self.k = min(k, self.cache_size)
        
        self.n_protos = n_protos
        self.proto_m = proto_m
        self.register_buffer("protos", torch.randn(self.n_protos,feat_dim))
        self.protos = F.normalize(self.protos, dim=-1)
        
        # Initialize class counts for each prototype
        self.proto_class_counts = torch.zeros(self.n_protos, self.num_classes).cuda() # ADDED
        
        self.distribution_limit = 0
        
    def sinkhorn(self, features):
        out = torch.matmul(features, self.protos.detach().T)
            
        Q = torch.exp(out.detach() / self.epsilon).t()# Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if torch.isinf(sum_Q):
            self.protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, self.ws(self.protos.detach()).T)
            Q = torch.exp(out.detach() / self.epsilon).t()# Q is K-by-B for consistency with notations from our paper
            sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q = F.normalize(Q, dim=1, p=1)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q = F.normalize(Q, dim=0, p=1)
            Q /= B

        Q *= B
        return Q.t()
        
    def mle_loss(self, features, targets, update_prototypes=True):
        # update prototypes by EMA
        anchor_labels = targets.contiguous().repeat(self.nviews).view(-1, 1)
        contrast_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda()
                
        Q = self.sinkhorn(features)

        # topk
        if self.k > 0:
            update_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            update_mask = F.normalize(F.normalize(topk_mask*update_mask, dim=1, p=1),dim=0, p=1)
        # original
        else:
            update_mask = F.normalize(F.normalize(mask * Q, dim=1, p=1),dim=0, p=1)
        update_features = torch.matmul(update_mask.T, features)
        
        if update_prototypes:
            self.proto_class_counts += torch.matmul(update_mask.T, F.one_hot(targets, num_classes=self.num_classes).float()) # ADDED
            protos = self.protos
            protos = self.proto_m * protos + (1-self.proto_m) * update_features
            self.protos = F.normalize(protos, dim=1, p=2)
        
        Q = self.sinkhorn(features)
        
        proto_dis = torch.matmul(features, self.protos.detach().T)
        anchor_dot_contrast = torch.div(proto_dis, self.temp)
        logits = anchor_dot_contrast
       
        if self.k > 0:
            loss_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            loss_mask = F.normalize(topk_mask*loss_mask, dim=1, p=1)
            masked_logits = loss_mask * logits 
        else:  
            masked_logits = F.normalize(Q*mask, dim=1, p=1) * logits
    
        pos=torch.sum(masked_logits, dim=1)
        neg=torch.log(torch.sum(torch.exp(logits), dim=1, keepdim=True))
        log_prob=pos-neg
        
        loss = -torch.mean(log_prob)
        return loss   
    
    def proto_contra(self, features, targets, temperature=0.5):
        total_features = features.shape[0]
        
        # Create label mask
        feature_labels = targets.contiguous().repeat(self.nviews).view(-1,1)
        mask = torch.eq(feature_labels, feature_labels.T).float().cuda()
        
        # Compute similarity matrix with numerical stability fixes
        features = F.normalize(features, dim=1)  # Ensure features are normalized
        anchor_dot_contrast = torch.matmul(features, features.T)
        
        # Apply temperature scaling
        anchor_dot_contrast = anchor_dot_contrast / temperature
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(total_features).view(-1, 1).to('cuda'),
            0
        )
        mask = mask * logits_mask
        
        # Add numerical stability checks
        exp_logits = torch.exp(logits) * logits_mask
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
        
        # Compute pos term with mask sum clamp
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1e-6)  # Prevent division by zero
        pos = torch.sum(mask * logits, dim=1) / mask_sum
        
        # Compute final loss with stability checks
        log_prob = pos - log_sum_exp.squeeze()
        loss = -torch.mean(log_prob)
        return loss
    
    
    def predict(self, features):
        # Assign the majority class to each prototype based on class counts
        _, proto_classes = torch.max(self.proto_class_counts, dim=1)
        
        # Compute the similarity between input features and prototypes
        similarity = torch.matmul(features, self.protos.T)
        
        # Get the index of the prototype with the highest similarity
        distances, prototype_indices = torch.max(similarity, dim=1)
        
        # Map the prototype indices to their corresponding class labels
        predicted_classes = proto_classes[prototype_indices]
        
        # Convert similarity to distance
        distances = 2 - 2 * distances
        
        return predicted_classes, distances

    def forward(self, features, targets, update_prototypes=True):
        loss = 0
        loss_dict = {}

        g_con = self.mle_loss(features, targets, update_prototypes)
        loss += g_con
        loss_dict['mle'] = g_con.cpu().item()
        
        g_dis = 0
        if self.lambda_pcon > 0:            
            g_dis = self.lambda_pcon * self.proto_contra(features, targets)
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()
                                
        self.protos = self.protos.detach()
                
        return loss, loss_dict
    
    
    
    
    # Saving / Loading State
    def save_state(self, filename, max_distance = 0):
        state = {
            'protos': self.protos,
            'proto_class_counts': self.proto_class_counts,
            'num_classes': self.num_classes,
            'temp': self.temp,
            'nviews': self.nviews,
            'cache_size': self.cache_size,
            'lambda_pcon': self.lambda_pcon,
            'feat_dim': self.feat_dim,
            'epsilon': self.epsilon,
            'sinkhorn_iterations': self.sinkhorn_iterations,
            'k': self.k,
            'n_protos': self.n_protos,
            'proto_m': self.proto_m,
            'distribution_limit': max_distance
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        
    def load_state(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                state = pickle.load(f)
            
            # Update all the attributes
            for key, value in state.items():
                setattr(self, key, value)
                
            print(f"PALM state loaded")
        else:
            print(f"No palm checkpoint found: {filename}")
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
        self.protos = nn.Parameter(
            F.normalize(
                torch.randn(self.n_protos, feat_dim), 
                dim=-1
            ),
            requires_grad=True
        )
        
        # Initialize class counts for each prototype
        self.proto_class_counts = torch.zeros(self.n_protos, self.num_classes).cuda() # ADDED
        
        self.distribution_limit = 0
        
    def sinkhorn(self, features):
        # Compute similarity scores
        out = torch.matmul(features, F.normalize(self.protos, dim=1).T)
        
        # Apply numerical stability fixes
        out_max, _ = torch.max(out, dim=1, keepdim=True)
        out_scaled = out - out_max  # Subtract maximum for numerical stability
        
        # Compute Q with better numerical stability
        Q = torch.exp(out_scaled / self.epsilon).t()
        
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # Normalize Q
        sum_Q = torch.sum(Q)
        if torch.isinf(sum_Q) or torch.isnan(sum_Q):
            # If we still get numerical issues, fall back to an even more stable computation
            normalized_protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, normalized_protos.T)
            out_max, _ = torch.max(out, dim=1, keepdim=True)
            out_scaled = out - out_max
            Q = torch.exp(out_scaled / self.epsilon).t()
            sum_Q = torch.sum(Q)
        
        Q = Q / sum_Q

        for _ in range(self.sinkhorn_iterations):
            # Row normalization
            Q = F.normalize(Q, dim=1, p=1)
            Q = Q / K
            
            # Column normalization
            Q = F.normalize(Q, dim=0, p=1)
            Q = Q / B

        Q = Q * B
        return Q.t().clone()  # Use clone() to ensure we return a new tensor
            
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
        
        if update_prototypes:
            with torch.no_grad():
                self.proto_class_counts += torch.matmul(
                    Q.T, F.one_hot(targets, num_classes=self.num_classes).float()
                )

        proto_dis = torch.matmul(features, F.normalize(self.protos, dim=1).T)
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
    
    def proto_contra(self, proto_temp = 0.1):
        proto_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(proto_labels, proto_labels.T).float().cuda()    

        contrast_feature = F.normalize(self.protos, dim=1)  # Normalize here
        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            proto_temp)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(self.n_protos).view(-1, 1).to('cuda'),
            0
        )
        mask = mask*logits_mask
        
        pos = torch.sum(F.normalize(mask, dim=1, p=1)*logits, dim=1)
        neg=torch.log(torch.sum(logits_mask * torch.exp(logits), dim=1))
        log_prob=pos-neg

        # loss
        loss = - torch.mean(log_prob)
        return loss
    
    
    def predict(self, features):
        # Assign the majority class to each prototype based on class counts
        _, proto_classes = torch.max(self.proto_class_counts, dim=1)
        
        # Compute the similarity between input features and prototypes
        similarity = torch.matmul(features, F.normalize(self.protos, dim=1).T)
        
        # Get the index of the prototype with the highest similarity
        distances, prototype_indices = torch.max(similarity, dim=1)
        
        # Map the prototype indices to their corresponding class labels
        predicted_classes = proto_classes[prototype_indices]
        
        # Convert similarity to distance
        distances = 2 - 2 * distances
        
        return predicted_classes, distances

    def forward(self, features, targets, update_prototypes=True, unlabeled_features=None):
        loss = 0
        loss_dict = {}

        g_con = self.mle_loss(features, targets, update_prototypes)
        loss += g_con
        loss_dict['mle'] = g_con.cpu().item()
        
        g_dis = 0     
        if self.lambda_pcon > 0:            
            g_dis = self.lambda_pcon * self.proto_contra()
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()

        #self.protos = self.protos.detach()
                
        return loss, loss_dict
    
    
    
    
    # Saving / Loading State
    def save_state(self, filename, max_distance = 0):
        state = {
            'protos': self.protos.data,
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
            
            if 'protos' in state:
                self.protos = nn.Parameter(state['protos'])
                del state['protos']

            # Update all the attributes
            for key, value in state.items():
                setattr(self, key, value)
                
            print(f"PALM state loaded")
        else:
            print(f"No palm checkpoint found: {filename}")
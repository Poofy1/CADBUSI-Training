import torch
import torch.nn as nn
import torch.nn.functional as F

class Embeddingmodel(nn.Module):
    def __init__(self, encoder, nf, num_classes=1, efficient_net=False):
        super(Embeddingmodel, self).__init__()
        self.encoder = encoder
        self.efficient_net = efficient_net
        self.num_classes = num_classes
        self.nf = nf
        self.aggregator = DSMIL(input_size=self.nf, num_classes=num_classes)
        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        
        print(f'{nf} Feature map size')

    def forward(self, input, projector=False, pred_on=False):
        num_bags = len(input) # input = [bag #, image #, channel, height, width]
        all_images = torch.cat(input, dim=0).cuda()  # Concatenate all bags into a single tensor for batch processing

        # Calculate the embeddings for all images in one go
        feat = self.encoder(all_images)
        #print(feat.shape)
        if not self.efficient_net:
            # Max pooling
            #feat = torch.max(feat, dim=2).values
            #feat = torch.max(feat, dim=2).values
            feat = feat.view(feat.shape[0], -1)
        #print(feat.shape)
        if pred_on:
            # Split the embeddings back into per-bag embeddings
            split_sizes = [bag.size(0) for bag in input]
            h_per_bag = torch.split(feat, split_sizes, dim=0)
        
            logits = torch.empty(num_bags, self.num_classes).cuda()
            yhat_instances = []
            for i, h in enumerate(h_per_bag):
                # Receive four values from the aggregator
                yhat_bag, yhat_ins, _ = self.aggregator(h)
                logits[i] = yhat_bag
                yhat_instances.append(yhat_ins)
        else:
            logits = None
            yhat_instances = None
            
        if projector:
            feat = self.projector(feat)
            feat = nn.functional.normalize(feat, dim=1)
            

        return logits, yhat_instances, feat
    
    
class DSMIL(nn.Module):
    def __init__(self, input_size, num_classes, dropout_v=0.0, nonlinear=False, passing_v=False): # K, L, N
        super(DSMIL, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU(), nn.Linear(512, 512), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 512)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        self.instance_fc = nn.Linear(input_size, num_classes)
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(num_classes, num_classes, kernel_size=input_size)
    
    def reset_parameters(self):
        # Reset all the parameters in the model
        for module in self.modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                module.reset_parameters()

    def forward(self, feats): # N x K
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        # Create critical instance predictions
        c = self.instance_fc(feats)
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        #print(q_max.transpose(0, 1))
        inst_pred = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        print(inst_pred)
        inst_pred = F.softmax( inst_pred / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(inst_pred.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        bag_pred = self.fcc(B) # 1 x C x 1
        bag_pred = bag_pred.view(1, -1)
        
        
        
        # (NOT IN PAPER)
        # Apply sigmoid to bag-level predictions and instance-level predictions
        bag_pred = torch.sigmoid(bag_pred)
        inst_pred = torch.sigmoid(inst_pred)
        
        
        
        return bag_pred, inst_pred, B
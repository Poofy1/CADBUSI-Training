import torch
import torch.nn as nn
import torch.nn.functional as F


class Embeddingmodel(nn.Module):
    def __init__(self, encoder, nf, num_classes=1, efficient_net = False):
        super(Embeddingmodel, self).__init__()
        self.encoder = encoder
        self.efficient_net = efficient_net
        
        self.num_classes = num_classes
        self.nf = nf
        
        self.aggregator = Linear_Classifier(nf = self.nf, num_classes = num_classes)

    def forward(self, input, pred_on=False):
        num_bags = len(input) # input = [bag #, image #, channel, height, width]
        
        all_images = torch.cat(input, dim=0).cuda() # Concatenate all bags into a single tensor for batch processing
        # Shape: [Total images in all bags, channel, height, width]
        
        # Calculate the embeddings for all images in one go
        feat = self.encoder(all_images)
        
        if not self.efficient_net:
            # Max pooling
            feat = torch.max(feat, dim=2).values  
            feat = torch.max(feat, dim=2).values 
        
         
        if pred_on:
            # Split the embeddings back into per-bag embeddings
            split_sizes = [bag.size(0) for bag in input]
            h_per_bag = torch.split(feat, split_sizes, dim=0)
            logits = torch.empty(num_bags, self.num_classes).cuda()
            yhat_instances = []
            for i, h in enumerate(h_per_bag):
                # Receive four values from the aggregator
                yhat_bag, yhat_ins = self.aggregator(h)
                logits[i] = yhat_bag
                yhat_instances.append(yhat_ins)
        else:
            logits = None
            yhat_instances = None

        return logits, yhat_instances, feat
    
    
    
class Linear_Classifier(nn.Module):
    """Linear classifier"""
    def __init__(self, nf, num_classes=1, L=128):
        super(Linear_Classifier, self).__init__()
        self.fc = nn.Linear(nf, num_classes)
        
        
        # Attention mechanism components
        self.attention_V = nn.Sequential(
            nn.Linear(nf, L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(nf, L),
            nn.Sigmoid()
        )
        self.attention_W = nn.Sequential(
            nn.Linear(L, 1),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(nf, num_classes),
            nn.Sigmoid()
        )

    def forward(self, v):
        
        # Gated-attention mechanism
        #v = torch.max(v, dim=2).values  
        #v = torch.max(v, dim=2).values  
        
        A_V = self.attention_V(v)  # KxL
        A_U = self.attention_U(v)  # KxL
        instance_scores = self.attention_W(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(instance_scores, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, v)  # ATTENTION_BRANCHESxM

        Y_prob = self.fc(Z)
        
        return Y_prob, instance_scores


import torch
import torch.nn as nn
from fastai.vision.all import *
import torch.nn.functional as F
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class Embeddingmodel(nn.Module):
    def __init__(self, encoder, nf, num_classes=1, efficient_net=False):
        super(Embeddingmodel, self).__init__()
        self.encoder = encoder
        self.efficient_net = efficient_net
        self.num_classes = num_classes
        self.nf = nf
        self.aggregator = Saliency_Classifier(nf=self.nf, num_classes=num_classes)
        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        
        print(f'Feature Map Size: {nf}')

    def forward(self, input, projector=False, pred_on = False):
        num_bags = len(input) # input = [bag #, image #, channel, height, width]
        all_images = torch.cat(input, dim=0).cuda()  # Concatenate all bags into a single tensor for batch processing

        # Calculate the embeddings for all images in one go
        feat = self.encoder(all_images)
        if not self.efficient_net:
            # Max pooling
            #feat = torch.max(feat, dim=2).values
            #feat = torch.max(feat, dim=2).values
            
            # Adaptive average pooling
            adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            feat = adaptive_avg_pool(feat).squeeze()
            
            # Global average pooling
            #feat = torch.mean(feat, dim=(2, 3))

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
            
        if projector:
            feat = self.projector(feat)
            feat = F.normalize(feat, dim=1)
            

        return logits, yhat_instances, feat



class Linear_Classifier(nn.Module):
    """Linear classifier"""
    def __init__(self, nf, num_classes=1, L=256):
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
        
        
    def reset_parameters(self):
        # Reset the parameters of all the submodules in the Linear_Classifier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        
        
    def forward(self, v):
        
        A_V = self.attention_V(v)  # KxL
        A_U = self.attention_U(v)  # KxL
        instance_scores = self.attention_W(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(instance_scores, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, v)  # ATTENTION_BRANCHESxM

        Y_prob = self.fc(Z)
        
        instance_scores = torch.sigmoid(instance_scores.squeeze())
        
        return Y_prob, instance_scores


class Linear_Classifier2(nn.Module):
    """Linear classifier"""
    def __init__(self, nf, num_classes=1, L=256):
        super(Linear_Classifier2, self).__init__()
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
        
        
    def reset_parameters(self):
        # Reset the parameters of all the submodules in the Linear_Classifier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        
        
    def forward(self, v):
        
        A_V = self.attention_V(v)  # KxL
        A_U = self.attention_U(v)  # KxL
        instance_scores = self.attention_W(A_V * A_U)  # element wise multiplication
        A = torch.transpose(instance_scores, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        
        # Apply fc layer to feat-level features
        feat_predictions = self.fc(v)  # KxC
        
        # Aggregate instance-level predictions
        Y_prob = torch.mm(A, feat_predictions)  # ATTENTION_BRANCHESxC

        instance_scores = torch.sigmoid(instance_scores.squeeze())
        return Y_prob, instance_scores

class Linear_Classifier3(nn.Module):
    """Linear classifier"""
    def __init__(self, nf, num_classes=1, L=256):
        super(Linear_Classifier3, self).__init__()
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
            nn.Linear(2*L, 1),  # Concatenate A_V and A_U
        )
        self.fc = nn.Sequential(
            nn.Linear(nf, num_classes),
            nn.Sigmoid()
        )
    
    def reset_parameters(self):
        # Reset the parameters of all the submodules in the Linear_Classifier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
    
    def forward(self, v):
        A_V = self.attention_V(v)  # KxL
        A_U = self.attention_U(v)  # KxL
        concat_features = torch.cat((A_V, A_U), dim=1)  # Kx(2*L)
        instance_scores = self.attention_W(concat_features)  # Kx1
        
        A = torch.transpose(instance_scores, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        
        # Apply fc layer to feat-level features
        feat_predictions = self.fc(v)  # KxC
        
        # Aggregate instance-level predictions
        Y_prob = torch.mm(A, feat_predictions)  # ATTENTION_BRANCHESxC
        instance_scores = torch.sigmoid(instance_scores.squeeze())
        
        return Y_prob, instance_scores
    
class Saliency_Classifier(nn.Module):
    """Linear classifier"""
    def __init__(self, nf, num_classes=1, L=256):
        super(Saliency_Classifier, self).__init__()
        self.fc = nn.Linear(nf, num_classes)
        self.pool_patches = 3
        
        self.saliency_layer = nn.Sequential(        
            nn.Conv2d(nf, num_classes, (1,1), bias = False),
            nn.Sigmoid()
        )
        
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
            nn.Linear(L, num_classes),
        )
        
        
    def reset_parameters(self):
        # Reset the parameters of all the submodules in the Linear_Classifier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        
        
    def forward(self, h):

        saliency_maps = self.saliency_layer(h)  # Generate saliency maps using a convolutional layer
        map_flatten = saliency_maps.flatten(start_dim=-2, end_dim=-1) 
        print(map_flatten.shape)
        # Select top patches based on saliency
        selected_area = map_flatten.topk(self.pool_patches, dim=2)[0]
        yhat_instance = selected_area.mean(dim=2).squeeze()  # Calculate the mean of the selected patches for instance predictions

        # Gated-attention mechanism
        v = torch.max(h, dim=2).values  # Max pooling across one dimension
        v = torch.max(v, dim=2).values  # Max pooling across the remaining spatial dimension
        A_V = self.attention_V(v)  # Learn attention features with a linear layer and Tanh activation
        A_U = self.attention_U(v)  # Learn gating mechanism with a linear layer and Sigmoid activation
        
        # Compute pre-softmax attention scores
        pre_softmax_scores = self.attention_W(A_V * A_U)

        # Apply softmax across the correct dimension (assuming the last dimension represents instances)
        attention_scores = nn.functional.softmax(pre_softmax_scores.squeeze() , dim=0)
        
        # Aggregate individual predictions to get the final bag prediction
        yhat_bag = (attention_scores * yhat_instance).sum(dim=0)
        return yhat_bag, yhat_instance
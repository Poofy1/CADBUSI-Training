import torch
import torch.nn as nn
from fastai.vision.all import *
import torch.nn.functional as F
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128):
        super(Embeddingmodel, self).__init__()
        
        # Get Head
        self.is_efficientnet = "efficientnet" in arch.lower()
        
        if self.is_efficientnet:
            self.encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            nf = 512
            # Replace the last fully connected layer with a new one
            num_features = self.encoder.classifier[1].in_features
            #self.encoder.classifier[1] = nn.Linear(num_features, nf)
            self.encoder.classifier[1] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, nf)
            )
        else:
            self.encoder = create_timm_body(arch, pretrained=pretrained_arch)
            nf = num_features_model(nn.Sequential(*self.encoder.children()))
            
        
        self.num_classes = num_classes
        self.nf = nf

        self.aggregator = Linear_Classifier(nf=self.nf, num_classes=num_classes)
        dropout_rate=0.5
        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feat_dim)
        )
        
        self.ins_classifier = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Add dropout after the first ReLU
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'Feature Map Size: {nf}')

    def forward(self, input, projector=False, pred_on = False):
        if pred_on:
            num_bags = len(input) # input = [bag #, image #, channel, height, width]
            all_images = torch.cat(input, dim=0).cuda()  # Concatenate all bags into a single tensor for batch processing
        else:
            all_images = input

        # Calculate the embeddings for all images in one go
        feat = self.encoder(all_images)
        if not self.is_efficientnet:
            feat = self.adaptive_avg_pool(feat).squeeze()
        
        
        # INSTANCE CLASS
        instance_predictions = self.ins_classifier(feat)

        bag_pred = None
        bag_instance_predictions = None
        if pred_on:
            # Split the embeddings back into per-bag embeddings
            split_sizes = [bag.size(0) for bag in input]
            h_per_bag = torch.split(feat, split_sizes, dim=0)
            y_hat_per_bag = torch.split(instance_predictions, split_sizes, dim=0)
            bag_pred = torch.empty(num_bags, self.num_classes).cuda()
            bag_instance_predictions = []
            for i, (h, y_h) in enumerate(zip(h_per_bag, y_hat_per_bag)):
                # Pass both h and y_hat to the aggregator
                yhat_bag, yhat_ins = self.aggregator(h, y_h)
                bag_pred[i] = yhat_bag
                bag_instance_predictions.append(yhat_ins) 
        
        proj = None
        if projector:
            proj = self.projector(feat)
            proj = F.normalize(proj, dim=1)
            

        return bag_pred, bag_instance_predictions, instance_predictions.squeeze(), proj


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
        
    def reset_parameters(self):
        # Reset the parameters of all the submodules in the Linear_Classifier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        
        
    def forward(self, v, y_hat):
        
        A_V = self.attention_V(v)  # KxL
        A_U = self.attention_U(v)  # KxL
        instance_scores = self.attention_W(A_V * A_U)  # element wise multiplication
        A = torch.transpose(instance_scores, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        # Aggregate instance-level predictions
        Y_prob = torch.mm(A, y_hat)  # ATTENTION_BRANCHESxC

        instance_scores = torch.sigmoid(instance_scores.squeeze())
        return Y_prob, instance_scores
    
    
    
class Saliency_Classifier(nn.Module):
    """Linear classifier"""
    def __init__(self, nf, num_classes=1, L=256):
        super(Saliency_Classifier, self).__init__()
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
            nn.Linear(L, num_classes),
        )
        
        
    def reset_parameters(self):
        # Reset the parameters of all the submodules in the Linear_Classifier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        
        
    def forward(self, h, yhat_instance):
        
        # Gated-attention mechanism
        v = torch.max(h, dim=2).values  # Max pooling across one dimension
        v = torch.max(v, dim=2).values  # Max pooling across the remaining spatial dimension
        A_V = self.attention_V(v)  # Learn attention features with a linear layer and Tanh activation
        A_U = self.attention_U(v)  # Learn gating mechanism with a linear layer and Sigmoid activation
        
        # Compute pre-softmax attention scores
        pre_softmax_scores = self.attention_W(A_V * A_U)
        pre_softmax_scores += 1e-7 # Added stability

        # Apply softmax across the correct dimension (assuming the last dimension represents instances)
        attention_scores = nn.functional.softmax(pre_softmax_scores.squeeze(), dim=0)
        
        # Aggregate individual predictions to get the final bag prediction
        yhat_bag = (attention_scores * yhat_instance).sum(dim=0)
        yhat_bag = torch.clamp(yhat_bag, min=1e-6, max=1-1e-6)
        return yhat_bag, yhat_instance
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
            self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            nf = 512
            # Replace the last fully connected layer with a new one
            num_features = self.encoder.classifier[1].in_features
            #self.encoder.classifier[1] = nn.Linear(num_features, nf)
            self.encoder.classifier[1] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, nf)
            )
        else:
            self.encoder = create_timm_body(arch, pretrained=pretrained_arch)
            nf = num_features_model(nn.Sequential(*self.encoder.children()))
            
        
        self.num_classes = num_classes
        self.nf = nf

        self.aggregator = MultiHeadAttention(nf=self.nf)
        dropout_rate=0.2
        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feat_dim)
        )
        
        print(f'Feature Map Size: {nf}')

    def forward(self, input, projector=False, pred_on = False):
        if pred_on:
            num_bags = len(input) # input = [bag #, image #, channel, height, width]
            all_images = torch.cat(input, dim=0).cuda()  # Concatenate all bags into a single tensor for batch processing
        else:
            all_images = input

        # Calculate the embeddings for all images in one go
        feat = self.encoder(all_images)
        

        bag_pred = None
        bag_instance_predictions = None
        if pred_on:
            # Split the embeddings back into per-bag embeddings
            split_sizes = [bag.size(0) for bag in input]
            h_per_bag = torch.split(feat, split_sizes, dim=0)
            bag_pred = torch.empty(num_bags, self.num_classes).cuda()
            bag_instance_predictions = []
            for i, h in enumerate(h_per_bag):
                yhat_bag, yhat_ins = self.aggregator(h)
                bag_pred[i] = yhat_bag
                bag_instance_predictions.append(yhat_ins) 
        
        proj = None
        if projector:
            proj = self.projector(feat)
            proj = F.normalize(proj, dim=1)
        
        # Clean up large intermediate tensors
        del feat
        if pred_on:
            del all_images
                

        return bag_pred, bag_instance_predictions, None, proj


class MultiHeadAttention(nn.Module):
    def __init__(self, nf, num_heads=4, L=256):
        super().__init__()
        self.num_heads = num_heads
        
        # Create separate attention components for each head
        self.attention_V = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nf, L),
                nn.Tanh()
            ) for _ in range(num_heads)
        ])
        
        self.attention_U = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nf, L),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])
        
        self.attention_W = nn.ModuleList([
            nn.Linear(L, 1) for _ in range(num_heads)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(nf * num_heads, 1),
            nn.Sigmoid()
        )

    def forward(self, v):
        all_weighted_features = []
        all_instance_scores = []
        
        for head in range(self.num_heads):
            A_V = self.attention_V[head](v)
            A_U = self.attention_U[head](v)
            instance_scores = self.attention_W[head](A_V * A_U)
            
            A = torch.transpose(instance_scores, 1, 0)
            A = F.softmax(A, dim=1)
            
            weighted_features = torch.mm(A, v)
            all_weighted_features.append(weighted_features)
            all_instance_scores.append(torch.sigmoid(instance_scores.squeeze()))
        
        # Concatenate features from all heads
        combined_features = torch.cat(all_weighted_features, dim=1)
        Y_prob = self.classifier(combined_features)
        
        # Average instance scores across heads
        final_instance_scores = torch.stack(all_instance_scores).mean(0)
        
        return Y_prob, final_instance_scores
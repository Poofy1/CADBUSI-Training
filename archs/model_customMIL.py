import torch
import torch.nn as nn
from fastai.vision.all import *
import torch.nn.functional as F
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from archs.linear_classifier import Linear_Classifier

class AttentionInstanceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, dropout_rate=0.2):
        super(AttentionInstanceClassifier, self).__init__()
        
        self.attention_dim = 512
        self.num_heads = num_heads
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, self.attention_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.attention_dim, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Add position encoding and reshape for attention
        x = x.unsqueeze(1) + self.pos_encoding  # [batch_size, 1, input_dim]
        
        # Apply multi-head attention
        attended_feat, _ = self.attention(x, x, x)
        
        # Squeeze the sequence dimension and classify
        attended_feat = attended_feat.squeeze(1)
        return self.classifier(attended_feat)


class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128):
        super(Embeddingmodel, self).__init__()
        
        # Get Head
        self.is_efficientnet = "efficientnet" in arch.lower()
        
        if self.is_efficientnet:
            self.encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            nf = 512
            num_features = self.encoder.classifier[1].in_features
            self.encoder.classifier[1] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, nf)
            )
        else:
            self.encoder = create_timm_body(arch, pretrained=pretrained_arch)
            nf = num_features_model(nn.Sequential(*self.encoder.children()))
        
        self.num_classes = num_classes
        self.nf = nf

        self.aggregator = Linear_Classifier(nf=self.nf, num_classes=num_classes)
        dropout_rate = 0.2
        
        # Instance classifier with attention
        self.ins_classifier = AttentionInstanceClassifier(
            input_dim=nf,
            num_classes=num_classes,
            num_heads=4,
            dropout_rate=dropout_rate
        )
        
        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feat_dim)
        )
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'Feature Map Size: {nf}')

    def forward(self, input, projector=False, pred_on=False):
        if pred_on:
            num_bags = len(input)
            all_images = torch.cat(input, dim=0).cuda()
        else:
            all_images = input

        # Calculate embeddings
        feat = self.encoder(all_images)
        if not self.is_efficientnet:
            feat = self.adaptive_avg_pool(feat).squeeze()
        
        # Get instance predictions using attention classifier
        instance_predictions = self.ins_classifier(feat)

        bag_pred = None
        bag_instance_predictions = None
        if pred_on:
            split_sizes = [bag.size(0) for bag in input]
            h_per_bag = torch.split(feat, split_sizes, dim=0)
            y_hat_per_bag = torch.split(instance_predictions, split_sizes, dim=0)
            bag_pred = torch.empty(num_bags, self.num_classes).cuda()
            bag_instance_predictions = []
            for i, (h, y_h) in enumerate(zip(h_per_bag, y_hat_per_bag)):
                yhat_bag, yhat_ins = self.aggregator(h, y_h)
                bag_pred[i] = yhat_bag
                bag_instance_predictions.append(yhat_ins)
        
        proj = None
        if projector:
            proj = self.projector(feat)
            proj = F.normalize(proj, dim=1)

        return bag_pred, bag_instance_predictions, instance_predictions.squeeze(), proj

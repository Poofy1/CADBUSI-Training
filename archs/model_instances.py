import torch
import torch.nn as nn
from fastai.vision.all import *
import torch.nn.functional as F
import timm
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128):
        super(Embeddingmodel, self).__init__()
        
        # 1. Use EfficientNet-V2 instead of original EfficientNet
        if "efficientnet" in arch.lower():
            self.encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
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
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(nf),  # Normalize features
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
        
        # 4. Label smoothing loss instead of regular BCE
        self.label_smoothing = 0.1
        
        # 5. Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, projector=False, pred_on=False):
        # Extract features
        features = self.encoder(x)
        
        # Get predictions
        predictions = self.classifier(features)
        
        return None, None, predictions.squeeze(), features

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)
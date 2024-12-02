import torch
import torch.nn as nn
from fastai.vision.all import *
from archs.backbone import create_timm_body

class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128):
        super(Embeddingmodel, self).__init__()
        

        self.encoder = create_timm_body('resnet10t', pretrained=False)
        nf = num_features_model(nn.Sequential(*self.encoder.children()))
            
        self.num_classes = num_classes
        self.nf = nf
        print(f"nf: {nf}")
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(nf, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
        

    def forward(self, x, projector=False, pred_on=False):
        # Extract features
        features = self.encoder(x)
        
        # Pool and flatten the features
        features = self.pool(features)
        features = features.view(features.size(0), -1)

        # Get predictions
        predictions = self.classifier(features)
        
        return None, None, predictions.squeeze(), features
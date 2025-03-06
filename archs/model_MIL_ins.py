import torch
import torch.nn as nn
from fastai.vision.all import *
import torch.nn.functional as F
from archs.backbone import *
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from archs.linear_classifier import *


class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128):
        super(Embeddingmodel, self).__init__()
        
        # Get Head
        self.is_efficientnet = "efficientnet" in arch.lower()
        
        if self.is_efficientnet:
            self.encoder = get_efficientnet_model(arch, pretrained_arch) 
            self.nf = get_num_features(self.encoder)
        else:
            base_encoder = create_timm_body(arch, pretrained=pretrained_arch)
            self.encoder = nn.Sequential(
                base_encoder,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            self.nf = num_features_model(nn.Sequential(*self.encoder.children()))
            
        
        self.num_classes = num_classes

        self.aggregator = Linear_Classifier(nf=self.nf, num_classes=num_classes)
        #self.aggregator = Saliency_Classifier(nf=self.nf, num_classes=num_classes)
        dropout_rate=0.2
        self.projector = nn.Sequential(
            nn.Linear(self.nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feat_dim)
        )
        
        self.ins_classifier = nn.Sequential(
            nn.Linear(self.nf, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Add dropout after the first ReLU
            nn.Linear(256, num_classes),
        )
        
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'Feature Map Size: {self.nf}')

    def forward(self, bags, projector=False, pred_on = False):
        if pred_on:
            num_bags = len(bags)  # input = [bag #, images_per_bag (padded), 224, 224, 3]
        
            all_images = []
            split_sizes = []
            
            for bag in bags:
                # Remove padded images (assuming padding is represented as zero tensors)
                valid_images = bag[~(bag == 0).all(dim=1).all(dim=1).all(dim=1)] # Shape: [valid_images, 224, 224, 3]
                
                split_sizes.append(valid_images.size(0))  # Track original bag sizes
                all_images.append(valid_images)
            
            if len(all_images) == 0:
                return None, None  # Handle case where no valid images exist
            
            all_images = torch.cat(all_images, dim=0)  # Shape: [Total valid images, 224, 224, 3]
        else:
            all_images = bags
        
        
        # Forward pass through encoder
        feats = self.encoder(all_images)  
        if len(feats.shape) == 4:
            feats = self.adaptive_avg_pool(feats).squeeze(-1).squeeze(-1) # Output shape: [Total valid images, feature_dim]
        
        
        # INSTANCE CLASS
        instance_predictions = self.ins_classifier(feats)

        bag_pred = None
        bag_instance_predictions = None
        if pred_on:
            # Split the embeddings back into per-bag embeddings
            h_per_bag = torch.split(feats, split_sizes, dim=0)
            y_hat_per_bag = torch.split(instance_predictions, split_sizes, dim=0)
            bag_pred = torch.empty(num_bags, self.num_classes, device=feats.device)
            bag_instance_predictions = []
            for i, (h, y_h) in enumerate(zip(h_per_bag, y_hat_per_bag)):
                # Pass both h and y_hat to the aggregator
                yhat_bag, yhat_ins = self.aggregator(h, y_h)
                bag_pred[i] = yhat_bag
                bag_instance_predictions.append(yhat_ins) 
                
            bag_instance_predictions = torch.cat(bag_instance_predictions, dim=0).squeeze()
        
        proj = None
        if projector:
            proj = self.projector(feats)
            proj = F.normalize(proj, dim=1)
        
        # Clean up large intermediate tensors
        del feats
        if pred_on:
            del all_images
            
        
                
        return bag_pred, bag_instance_predictions, instance_predictions.squeeze(), proj
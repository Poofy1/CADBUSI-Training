import torch
import torch.nn as nn
from fastai.vision.all import *
import torch.nn.functional as F
from archs.backbone import *
from archs.linear_classifier import *


class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128):
        super(Embeddingmodel, self).__init__()
        
        # Get Head
        if "efficientnet" in arch.lower():
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

        self.aggregator = Linear_Classifier_With_FC(nf=self.nf, num_classes=num_classes)
        dropout_rate=0.2
        self.projector = nn.Sequential(
            nn.Linear(self.nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feat_dim)
        )
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'Feature Map Size: {self.nf}')

    def forward(self, bags, projector=False, pred_on = False):
        if pred_on:
            num_bags = len(bags)
            bag_pred = torch.empty(num_bags, self.num_classes, device=bags.device)
            bag_instance_predictions = []
            
            # Process each bag separately to avoid large concatenation
            for i, bag in enumerate(bags):
                # Remove padded images
                valid_images = bag[~(bag == 0).all(dim=1).all(dim=1).all(dim=1)]
                if valid_images.size(0) == 0:
                    continue
                    
                # Process just this bag
                feats = self.encoder(valid_images)
                if len(feats.shape) == 4:
                    feats = self.adaptive_avg_pool(feats).squeeze(-1).squeeze(-1)
                    
                # Get predictions for this bag
                yhat_bag, yhat_ins = self.aggregator(feats)
                bag_pred[i] = yhat_bag
                bag_instance_predictions.append(yhat_ins)
                

            return bag_pred, None, bag_instance_predictions, None
        else:
            all_images = bags
            split_sizes = [1] * bags.size(0)
            num_bags = bags.size(0)
        
        # Forward pass through encoder
        feats = self.encoder(all_images)  
        if len(feats.shape) == 4:
            feats = self.adaptive_avg_pool(feats).squeeze(-1).squeeze(-1) # Output shape: [Total valid images, feature_dim]
        
        bag_pred = None
        bag_instance_predictions = None

        # Split the embeddings back into per-bag embeddings
        h_per_bag = torch.split(feats, split_sizes, dim=0)
        bag_pred = torch.empty(num_bags, self.num_classes, device=feats.device)
        bag_instance_predictions = []
        for i, h in enumerate(h_per_bag):
            # Pass both h and y_hat to the aggregator
            yhat_bag, yhat_ins = self.aggregator(h)
            bag_pred[i] = yhat_bag
            bag_instance_predictions.append(yhat_ins) 
        
        
        
        proj = None
        if projector:
            proj = self.projector(feats)
            proj = F.normalize(proj, dim=1)
        
        # Clean up large intermediate tensors
        del feats
        if pred_on:
            del all_images
            
        
        if not pred_on:
            bag_instance_predictions = torch.stack(bag_instance_predictions)
            
            
        return bag_pred, None, bag_instance_predictions, proj
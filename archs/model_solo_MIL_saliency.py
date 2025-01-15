import torch
import torch.nn as nn
from fastai.vision.all import *
import torch.nn.functional as F
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from archs.linear_classifier import *

class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128):
        super(Embeddingmodel, self).__init__()
        
        # Get Head
        self.is_efficientnet = "efficientnet" in arch.lower()
        
        if self.is_efficientnet:
            self.encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            nf = 1536  # EfficientNet-B3's feature map has 1536 channels
            # Remove the classifier to keep spatial dimensions
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        else:
            self.encoder = create_timm_body(arch, pretrained=pretrained_arch)
            nf = num_features_model(nn.Sequential(*self.encoder.children()))
            
        
        self.num_classes = num_classes
        self.nf = nf

        self.aggregator = Linear_Classifier(nf=self.nf, num_classes=num_classes)
        dropout_rate=0.2
        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feat_dim)
        )
        
        self.saliency_layer = nn.Sequential(
            nn.Conv2d(nf, num_classes, (1,1), bias = False),
            nn.Sigmoid()
        )
        
        """self.saliency_layer = nn.Sequential(
            nn.Conv2d(nf, nf//2, 3, padding=1),
            nn.BatchNorm2d(nf//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf//2, nf//4, 3, padding=1),
            nn.BatchNorm2d(nf//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf//4, num_classes, 1),
            nn.Sigmoid()
        )"""

        self.pool_patches = 3
        
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
        
        
        # SALIENCY CLASS
        saliency_maps = self.saliency_layer(feat)  # Generate saliency maps using a convolutional layer
        map_flatten = saliency_maps.flatten(start_dim=-2, end_dim=-1) 
        selected_area = map_flatten.topk(self.pool_patches, dim=2)[0]
        instance_predictions = selected_area.mean(dim=2).squeeze()  # Calculate the mean of the selected patches for instance predictions

        feat = self.adaptive_avg_pool(feat).squeeze()
    

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
                y_h = y_h.view(-1, 1)
                yhat_bag, yhat_ins = self.aggregator(h, y_h)
                bag_pred[i] = yhat_bag
                bag_instance_predictions.append(yhat_ins) 
        
        proj = None
        if projector:
            proj = self.projector(feat)
            proj = F.normalize(proj, dim=1)
        else:
            proj = saliency_maps
            

        return bag_pred, bag_instance_predictions, instance_predictions.squeeze(), proj
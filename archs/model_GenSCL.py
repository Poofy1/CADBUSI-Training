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
            nf = 512
            # Replace the last fully connected layer with a new one
            num_features = self.encoder.classifier[1].in_features
            self.encoder.classifier[1] = nn.Linear(num_features, nf)
        else:
            base_encoder = create_timm_body(arch, pretrained=pretrained_arch)
            self.encoder = nn.Sequential(
                base_encoder,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            nf = num_features_model(base_encoder)
            
            
        self.num_classes = num_classes
        self.nf = nf
        
        #self.aggregator = Saliency_Classifier(nf=self.nf, num_classes=num_classes)
        self.aggregator = Linear_Classifier_With_FC(nf=self.nf, num_classes=num_classes)

        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'Feature Map Size: {nf}')

    def forward(self, input, projector=False, pred_on = False):
        num_bags = len(input) # input = [bag #, image #, channel, height, width]
        all_images = torch.cat(input, dim=0).cuda()  # Concatenate all bags into a single tensor for batch processing

        # Calculate the embeddings for all images in one go
        feat = self.encoder(all_images)

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
            #feat = self.adaptive_avg_pool(feat).squeeze()
            feat = self.projector(feat)
            feat = F.normalize(feat, dim=1)
            
        # Clean up large intermediate tensors
        if pred_on:
            del all_images
            
        return logits, yhat_instances, None, feat

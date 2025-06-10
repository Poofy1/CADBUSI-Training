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
        self.is_efficientnet = "efficientnet" in arch.lower()
        
        if self.is_efficientnet:
            model = get_efficientnet_model(arch, pretrained_arch) 
            self.nf = get_num_features(model)
            self.encoder, pooled_size = create_pooled_efficientnet(model)
                        
        else:
            #self.encoder, pooled_size = create_pooled_resnet(arch, pretrained=pretrained_arch)
            self.encoder, pooled_size = create_pooled_convnext(arch, pretrained=pretrained_arch)
            self.nf = num_features_model(nn.Sequential(*self.encoder.children()))


        self.num_classes = num_classes
        self.aggregator = Attention_Prediction_Aggregator(nf=self.nf, num_classes=num_classes)
        dropout_rate=0.2
        self.projector = nn.Sequential(
            nn.Linear(self.nf, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feat_dim)
        )
        
        self.saliency_layer = nn.Sequential(
            nn.Conv2d(pooled_size, num_classes, (1,1), bias = True),
            #nn.Sigmoid() # removed sigmoid for stabilization 
        )
        self.pool_patches = 3
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'Feature Map Size: {self.nf}')
        print(f'Pooled Feature Map Size: {pooled_size}')

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

        # Calculate the embeddings for all images in one go
        feats, pooled_feat = self.encoder(all_images)
        if len(feats.shape) == 4:
            feats = self.adaptive_avg_pool(feats).squeeze(-1).squeeze(-1) # Output shape: [Total valid images, feature_dim]
        
        
        # SALIENCY CLASS
        saliency_maps = self.saliency_layer(pooled_feat)  # Generate saliency maps using a convolutional layer
        map_flatten = saliency_maps.flatten(start_dim=-2, end_dim=-1) 
        selected_area = map_flatten.topk(self.pool_patches, dim=2)[0]
        instance_predictions = selected_area.mean(dim=2).squeeze()  # Calculate the mean of the selected patches for instance predictions

        bag_pred = None
        bag_instance_predictions = None
        if pred_on:
            # Split the embeddings back into per-bag embeddings
            h_per_bag = torch.split(feats, split_sizes, dim=0)
            y_hat_per_bag = torch.split(instance_predictions, split_sizes, dim=0)
            bag_pred = torch.empty(num_bags, self.num_classes)
            bag_instance_predictions = []
            for i, (h, y_h) in enumerate(zip(h_per_bag, y_hat_per_bag)):
                # Pass both h and y_hat to the aggregator
                y_h = y_h.view(-1, 1)
                yhat_bag, yhat_ins = self.aggregator(h, y_h)
                bag_pred[i] = yhat_bag
                bag_instance_predictions.append(yhat_ins) 
        
        proj = None
        if projector:
            proj = self.projector(feats)
            proj = F.normalize(proj, dim=1)
        else:
            proj = saliency_maps
            

        return bag_pred, saliency_maps, instance_predictions.squeeze(), proj
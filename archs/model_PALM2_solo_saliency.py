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
            nf = 1536  # EfficientNet-B3's feature map has 1536 channels
            # Remove the classifier to keep spatial dimensions
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        else:
            self.encoder = create_timm_body(arch, pretrained=pretrained_arch)
            nf = num_features_model(nn.Sequential(*self.encoder.children()))
            
        
        self.num_classes = num_classes
        self.nf = nf

        self.aggregator = Linear_Classifier(nf=self.nf, num_classes=num_classes)
        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )
        
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.saliency_layer = nn.Sequential(
            nn.Conv2d(nf, num_classes, (1,1), bias = False),
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
        
        
        # SALIENCY CLASS
        upsampled_feat = self.upsample(feat)
        saliency_maps = self.saliency_layer(upsampled_feat)  # Generate saliency maps using a convolutional layer
        map_flatten = saliency_maps.flatten(start_dim=-2, end_dim=-1) 
        selected_area = map_flatten.topk(3, dim=2)[0]
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
    
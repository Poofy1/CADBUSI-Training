import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.backbone import *
from archs.linear_classifier import *
from archs.dsmil import *

class UnifiedAttentionAggregator(nn.Module):
    def __init__(self, nf, num_classes=1):
        super(UnifiedAttentionAggregator, self).__init__()
        self.num_classes = num_classes
        
        # Select Aggregator
        #self.aggregator = Attention_Prediction_Aggregator(nf, num_classes) # Includes seperate instance classifier 
        #self.aggregator = Attention_Feature_Classifier(nf, num_classes)
        self.aggregator = DSMIL(nf, num_classes)
        self.has_ins_classifier = hasattr(self.aggregator, 'ins_classifier')
        
    def reset_parameters(self):
        """Reset parameters of the underlying aggregator"""
        for module in self.aggregator.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                module.reset_parameters()
            elif isinstance(module, nn.BatchNorm1d):
                module.reset_parameters()
            elif isinstance(module, nn.BatchNorm2d):
                module.reset_parameters()
    
    def forward(self, feats, split_sizes, pred_on=True):
        num_bags = len(split_sizes)
        h_per_bag = torch.split(feats, split_sizes, dim=0)
        bag_pred = torch.empty(num_bags, self.num_classes, device=feats.device)
        
        if self.has_ins_classifier:
            # prediction aggregator logic
            instance_predictions = self.aggregator.ins_classifier(feats)
            if pred_on:
                y_hat_per_bag = torch.split(instance_predictions, split_sizes, dim=0)
                for i, (h, y_h) in enumerate(zip(h_per_bag, y_hat_per_bag)):
                    bag_pred[i] = self.aggregator(h, y_h)
            return bag_pred, instance_predictions.squeeze()
        else:
            # feature classifier logic
            for i, h in enumerate(h_per_bag):
                bag_pred[i] = self.aggregator(h)
            return bag_pred, None
            

class Embeddingmodel(nn.Module):
    def __init__(self, arch, pretrained_arch, num_classes=1, feat_dim=128, use_float_input=False):
        super(Embeddingmodel, self).__init__()
        
        # Get Head
        self.is_efficientnet = "efficientnet" in arch.lower()
        self.use_float_input = use_float_input
        
        if self.is_efficientnet:
            base_encoder = get_efficientnet_model(arch, pretrained_arch) 
            self.encoder = nn.Sequential(
                base_encoder,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            nf = num_features_model(nn.Sequential(*self.encoder.children()))
        else:
            base_encoder = create_timm_body(arch, pretrained=pretrained_arch)
            self.encoder = nn.Sequential(
                base_encoder,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            nf = num_features_model(nn.Sequential(*self.encoder.children()))
        
        # Add 1 to nf if using float input
        if self.use_float_input:
            nf += 1
            
        self.aggregator = UnifiedAttentionAggregator(nf=nf, num_classes=num_classes)
        
        self.projector = nn.Sequential(
            nn.Linear(nf, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        print(f'Feature Map Size: {nf}')

    def reset_aggregator_parameters(self):
        self.aggregator.reset_parameters()
        
    def forward(self, bags, float_input=None, projector=False, pred_on=False):
        
        # Calculate the embeddings for all images in one go
        if isinstance(bags, (list, tuple)):
            all_images = torch.cat(bags, dim=0).cuda()
            split_sizes = [bag.size(0) for bag in bags]
        else: 
            all_images = bags
            split_sizes = [bags.size(0)]
            
        feat = self.encoder(all_images)
        
        # Add float input to features if provided
        if self.use_float_input and float_input is not None:
            batch_size = feat.size(0)
            
            if isinstance(float_input, list):
                # Handle nested list structure (matching bags structure)
                if isinstance(float_input[0], list):
                    # Flatten the nested list and concatenate tensors
                    all_float_tensors = []
                    for bag_floats in float_input:
                        for float_tensor in bag_floats:
                            all_float_tensors.append(float_tensor)
                    float_tensor = torch.cat(all_float_tensors, dim=0).cuda()
                else:
                    # Handle single-level list of tensors
                    float_tensor = torch.cat(float_input, dim=0).cuda()
                
                # Reshape to match batch size
                float_tensor = float_tensor.view(batch_size, -1)
                
            elif isinstance(float_input, torch.Tensor):
                # Handle tensor input
                float_tensor = float_input.view(-1, 1).expand(batch_size, -1)
                
            else:
                # Handle single float/int value
                float_tensor = torch.full((batch_size, 1), float_input, device=feat.device)
            
            # Concatenate float input to features
            feat = torch.cat([feat, float_tensor], dim=1)

        # Get bag pred
        bag_pred, instance_pred = self.aggregator(feat, split_sizes, pred_on=pred_on)
            
        if projector:
            #feat = self.adaptive_avg_pool(feat).squeeze()
            feat = self.projector(feat)
            feat = F.normalize(feat, dim=1)
            
        # Clean up large intermediate tensors
        if pred_on:
            del all_images
        
        return bag_pred.cuda(), instance_pred, feat

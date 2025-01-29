from fastai.vision.all import *
from torch import nn
from archs.backbone import create_timm_body
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from archs.linear_classifier import *
torch.backends.cudnn.benchmark = True

class Embeddingmodel(nn.Module):
    
    def __init__(self, arch, pretrained_arch, num_classes=1):
        super(Embeddingmodel,self).__init__()
        # Get Head
        self.is_efficientnet = "efficientnet" in arch.lower()
        
        if self.is_efficientnet:
            self.encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            nf = 512
            # Replace the last fully connected layer with a new one
            num_features = self.encoder.classifier[1].in_features
            self.encoder.classifier[1] = nn.Linear(num_features, nf)
        else:
            self.encoder = create_timm_body(arch, pretrained=pretrained_arch)
            nf = num_features_model(nn.Sequential(*self.encoder.children()))
            
            
        self.aggregator = Linear_Classifier_With_FC(nf, num_classes=num_classes)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_classes = num_classes
        print(f'Feature Map Size: {nf}')

    def forward(self, bags):
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
        
        # Forward pass through encoder
        all_images = torch.cat(all_images, dim=0).cuda()  # Shape: [Total valid images, 224, 224, 3]
        feats = self.encoder(all_images)  
        feats = self.adaptive_avg_pool(feats).squeeze(-1).squeeze(-1) # Output shape: [Total valid images, feature_dim]
        
        # Split the embeddings back into per-bag groups
        h_per_bag = torch.split(feats, split_sizes, dim=0)

        logits = torch.empty(num_bags, self.num_classes).cuda()
        yhat_instances = []
        
        for i, h in enumerate(h_per_bag):
            # Pass bag features through aggregator
            yhat_bag, yhat_ins = self.aggregator(h)
            logits[i] = yhat_bag
            yhat_instances.append(yhat_ins)
        
        return logits, yhat_instances, None, None

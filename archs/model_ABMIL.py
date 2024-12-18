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
            
            
        self.aggregator = Linear_Classifier(nf, num_classes=num_classes)
        self.num_classes = num_classes
        print(f'Feature Map Size: {nf}')

    def forward(self, input):
        num_bags = len(input) # input = [bag #, image #, channel, height, width]
        
        # Concatenate all bags into a single tensor for batch processing
        all_images = torch.cat(input, dim=0)  # Shape: [Total images in all bags, channel, height, width]
        
        # Calculate the embeddings for all images in one go
        h_all = self.encoder(all_images.cuda())
        
        # Split the embeddings back into per-bag embeddings
        split_sizes = [bag.size(0) for bag in input]
        h_per_bag = torch.split(h_all, split_sizes, dim=0)
        logits = torch.empty(num_bags, self.num_classes).cuda()
        yhat_instances= []
        
        for i, h in enumerate(h_per_bag):
            # Receive four values from the aggregator
            yhat_bag, yhat_ins = self.aggregator(h)
            logits[i] = yhat_bag
            yhat_instances.append(yhat_ins)
        
        return logits, yhat_instances
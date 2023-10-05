from timm import create_model
from fastai.vision.all import *
from torch import nn
torch.backends.cudnn.benchmark = True


class ABMIL_aggregate(nn.Module):
    
    def __init__(self, nf, num_classes, pool_patches = 3, L = 128):
        super(ABMIL_aggregate,self).__init__()
        self.nf = nf
        self.num_classes = num_classes # two for binary classification
        self.pool_patches = pool_patches # how many patches to use in predicting instance label
        self.L = L # number of latent attention features   
        
        self.saliency_layer = nn.Sequential(        
            nn.Conv2d( self.nf, self.num_classes, (1,1), bias = False),
            nn.Sigmoid() )
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.nf, self.L),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.nf, self.L),
            nn.Sigmoid()
        )

        self.attention_W = nn.Sequential(
            nn.Linear(self.L, self.num_classes),
        )
                    
    def forward(self, h):
        # input is a tensor with a bag of features, dim = bag_size x nf x h x w
    
        h = h.permute(0, 3, 1, 2)  # Now h has shape [1, 512, 10, 64]
    
        saliency_maps = self.saliency_layer(h)
        map_flatten = saliency_maps.flatten(start_dim = -2, end_dim = -1)
        selected_area = map_flatten.topk(self.pool_patches, dim=2)[0]
        yhat_instance = selected_area.mean(dim=2).squeeze()
        
        # gated-attention
        v = torch.max( h, dim = 2).values # begin maxpool
        v = torch.max( v, dim = 2).values # maxpool complete
        A_V = self.attention_V(v) 
        A_U = self.attention_U(v) 
        attention_scores = nn.functional.softmax(
            self.attention_W(A_V * A_U).squeeze(), dim = 0 )
        
        # aggreate individual predictions to get bag prediciton
        yhat_bag = (attention_scores * yhat_instance).sum(dim=0)
       
        return yhat_bag, saliency_maps, yhat_instance, attention_scores

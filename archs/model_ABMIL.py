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
            nn.Sigmoid()
        )
        
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
        
        saliency_maps = self.saliency_layer(h)  # Generate saliency maps using a convolutional layer
        map_flatten = saliency_maps.flatten(start_dim=-2, end_dim=-1) 
        
        # Select top patches based on saliency
        selected_area = map_flatten.topk(self.pool_patches, dim=2)[0]
        yhat_instance = selected_area.mean(dim=2).squeeze()  # Calculate the mean of the selected patches for instance predictions

        # Gated-attention mechanism
        v = torch.max(h, dim=2).values  # Max pooling across one dimension
        v = torch.max(v, dim=2).values  # Max pooling across the remaining spatial dimension
        A_V = self.attention_V(v)  # Learn attention features with a linear layer and Tanh activation
        A_U = self.attention_U(v)  # Learn gating mechanism with a linear layer and Sigmoid activation
        
        # Compute pre-softmax attention scores
        pre_softmax_scores = self.attention_W(A_V * A_U)

        # Apply softmax across the correct dimension (assuming the last dimension represents instances)
        attention_scores = nn.functional.softmax(pre_softmax_scores.squeeze() , dim=0)

        # Aggregate individual predictions to get the final bag prediction
        yhat_bag = (attention_scores * yhat_instance).sum(dim=0)
        return yhat_bag, saliency_maps, yhat_instance, attention_scores

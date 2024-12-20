import torch.nn as nn
import torch
import torch.nn.functional as F



class Linear_Classifier(nn.Module):
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
        
    def forward(self, x, instance_pred):
        
        A_V = self.attention_V(x)  # KxL
        A_U = self.attention_U(x)  # KxL
        instance_scores = self.attention_W(A_V * A_U)  # element wise multiplication
        A = torch.transpose(instance_scores, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        # Aggregate instance-level predictions
        Y_prob = torch.mm(A, instance_pred)  # ATTENTION_BRANCHESxC

        instance_scores = torch.sigmoid(instance_scores.squeeze())
        return Y_prob, instance_scores
    



class Linear_Classifier_With_FC(nn.Module):
    def __init__(self, nf, num_classes=1, L=256):
        super(Linear_Classifier_With_FC, self).__init__()
        
        # Use LayerNorm instead of BatchNorm1d
        self.input_norm = nn.LayerNorm(nf)
        
        # Feature transformation before attention
        self.feature_transform = nn.Sequential(
            nn.Linear(nf, nf),
            nn.LayerNorm(nf),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(nf, nf),  # New layer
            nn.LayerNorm(nf),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        
        # Attention mechanism components
        self.attention_V = nn.Sequential(
            nn.Linear(nf, L),
            nn.LayerNorm(L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(nf, L),
            nn.LayerNorm(L),
            nn.Sigmoid()
        )
        self.attention_W = nn.Sequential(
            nn.Linear(L, 1)
        )
        
        # Classifier with LayerNorm
        self.classifier = nn.Sequential(
            nn.Linear(nf, nf//2),
            nn.LayerNorm(nf//2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(nf//2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, v):
        # Normalize input features
        v = self.input_norm(v)
        
        # Transform features
        v = self.feature_transform(v)
        
        # Attention mechanism
        A_V = self.attention_V(v)  # KxL
        A_U = self.attention_U(v)  # KxL
        instance_scores = self.attention_W(A_V * A_U)  # element wise multiplication
        A = torch.transpose(instance_scores, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        
        # Weight the features then get bag prediction
        weighted_features = torch.mm(A, v)  # weighted average of features
        Y_prob = self.classifier(weighted_features)
        
        instance_scores = torch.sigmoid(instance_scores.squeeze())
        return Y_prob, instance_scores
    
    
    
    
    
class Saliency_Classifier(nn.Module):
    def __init__(self, nf, num_classes=1, L=256):
        super(Saliency_Classifier, self).__init__()
        self.fc = nn.Linear(nf, num_classes)
        self.pool_patches = 3
        
        self.saliency_layer = nn.Sequential(        
            nn.Conv2d(nf, num_classes, (1,1), bias = False),
            nn.Sigmoid()
        )
        
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
            nn.Linear(L, num_classes),
        )
        
        
    def reset_parameters(self):
        # Reset the parameters of all the submodules in the Linear_Classifier
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        
        
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
        pre_softmax_scores += 1e-7 # Added stability

        # Apply softmax across the correct dimension (assuming the last dimension represents instances)
        attention_scores = nn.functional.softmax(pre_softmax_scores.squeeze(), dim=0)
        
        # Aggregate individual predictions to get the final bag prediction
        yhat_bag = (attention_scores * yhat_instance).sum(dim=0)
        #yhat_bag = torch.clamp(yhat_bag, min=1e-6, max=1-1e-6)
        return yhat_bag, yhat_instance
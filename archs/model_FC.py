from fastai.vision.all import *
from torch import nn
torch.backends.cudnn.benchmark = True

class FC_aggregate_old(nn.Module):
    
    def __init__(self, nf, num_classes, L = 128, fc_layers=[256, 128], dropout = .5):
        super(FC_aggregate, self).__init__()
        self.nf = nf
        self.num_classes = num_classes
        self.L = L
        self.dropout = dropout

        # Fully connected layers
        fc_seq = []
        in_features = nf
        for out_features in fc_layers:
            fc_seq.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_features = out_features
        fc_seq.append(nn.Linear(in_features, num_classes))
        self.fc_layers = nn.Sequential(*fc_seq)

        # Attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(nf, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(nf, self.L),
            nn.Sigmoid()
        )
        self.attention_W = nn.Sequential(
            nn.Linear(self.L, self.num_classes),
        )

    def forward(self, h):
        # Flatten and apply fully connected layers
        h_flat = h.view(h.size(0), h.size(1), -1).mean(dim=2)  # Global Average Pooling
        yhat_instance = self.fc_layers(h_flat).squeeze()  # Shape: [num_instances, num_classes]

        # Gated-attention mechanism
        v = h.view(h.size(0), h.size(1), -1).max(dim=2).values
        A_V = self.attention_V(v)
        A_U = self.attention_U(v)
        pre_softmax_scores = self.attention_W(A_V * A_U)
        attention_scores = nn.functional.softmax(pre_softmax_scores.squeeze(), dim=0)

        # Aggregate instance predictions to get the final bag prediction
        yhat_bag = (attention_scores * yhat_instance).sum(dim=0)
        
        yhat_bag = torch.sigmoid(yhat_bag)

        return yhat_bag, yhat_instance, attention_scores
    
    
    
class FC_aggregate(nn.Module):
    
    def __init__(self, nf, num_bag_classes, num_instance_classes, L=128, fc_layers=[256, 128], dropout=0.5):
        super(FC_aggregate, self).__init__()
        self.nf = nf
        self.num_bag_classes = num_bag_classes
        self.num_instance_classes = num_instance_classes
        self.L = L
        self.dropout = dropout

        # Fully connected layers for bag-level predictions
        bag_fc_seq = []
        in_features = nf
        for out_features in fc_layers:
            bag_fc_seq.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_features = out_features
        bag_fc_seq.append(nn.Linear(in_features, num_bag_classes))
        self.bag_fc_layers = nn.Sequential(*bag_fc_seq)

        # Fully connected layers for instance-level predictions
        instance_fc_seq = []
        in_features = nf
        for out_features in fc_layers:
            instance_fc_seq.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_features = out_features
        instance_fc_seq.append(nn.Linear(in_features, num_instance_classes))
        self.instance_fc_layers = nn.Sequential(*instance_fc_seq)

        # Common attention mechanism for bag-level predictions
        self.attention_V = nn.Sequential(
            nn.Linear(nf, self.L),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(nf, self.L),
            nn.Sigmoid()
        )
        self.attention_W = nn.Sequential(
            nn.Linear(self.L, self.num_bag_classes),
        )

    def forward(self, h):
        # Flatten and apply fully connected layers for bag-level predictions
        h_flat = h.view(h.size(0), h.size(1), -1).mean(dim=2)
        yhat_bag = self.bag_fc_layers(h_flat)  # Shape: [num_instances, num_bag_classes]

        # Flatten and apply fully connected layers for instance-level predictions
        yhat_instance = self.instance_fc_layers(h_flat)  # Shape: [num_instances, num_instance_classes]

        # Bag-level gated-attention mechanism
        v = h.view(h.size(0), h.size(1), -1).max(dim=2).values
        A_V = self.attention_V(v)
        A_U = self.attention_U(v)
        pre_softmax_scores = self.attention_W(A_V * A_U)
        bag_attention_scores = nn.functional.softmax(pre_softmax_scores.squeeze(), dim=0)

        # Aggregate instance predictions to get the final bag prediction
        yhat_bag = (bag_attention_scores * yhat_bag).sum(dim=0)
        yhat_bag = torch.sigmoid(yhat_bag)

        return yhat_bag, yhat_instance
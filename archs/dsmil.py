import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


"""class IClassifier(nn.Module):
    def __init__(self, feature_size, output_class, dropout=0.3):
        super(IClassifier, self).__init__()    
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_size // 2, output_class)
        )
        
    def forward(self, feats):
        c = self.classifier(feats.view(feats.shape[0], -1))
        return c.squeeze(1) if c.shape[1] == 1 else c"""
    
    
class IClassifier(nn.Module):
    def __init__(self, feature_size, output_class):
        super(IClassifier, self).__init__()    
        
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, feats):
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return c.squeeze(1)

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        c = c.unsqueeze(1)
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class DSMIL(nn.Module):
    def __init__(self, input_size, output_class):
        super(DSMIL, self).__init__()
        
        self.b_classifier = BClassifier(input_size, output_class)
        self.ins_classifier = IClassifier(input_size, output_class)
        
    def forward(self, feats, inst):
        prediction_bag, A, B = self.b_classifier(feats, inst)

        return prediction_bag
from fastai.vision.all import *
from torch import nn
from nystrom_attention import NystromAttention
torch.backends.cudnn.benchmark = True


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            out, attn = self.attn(self.norm(x), return_attn=True)
            x = x + out
            return x, attn.detach()
        else:
            x = x + self.attn(self.norm(x))
            return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        
        # Compute the values of H and W based on the total number of elements
        total_elements = feat_token.numel()
        H = W = int((total_elements / (B * C)) ** 0.5)
    
        
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, dim_in, dim_hid, n_classes, **kwargs):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=dim_hid)
        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hid), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hid))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=dim_hid)
        self.layer2 = TransLayer(dim=dim_hid)
        self.norm = nn.LayerNorm(dim_hid)
        self._fc2 = nn.Linear(dim_hid, self.n_classes)

    def forward(self, X, **kwargs):

        #assert X.shape[0] == 1 # [1, n, 1024], single bag

        h = self._fc1(X) # [B, n, dim_hid]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, dim_hid]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        
        # Reshape h to be a 3D tensor
        h = h.view(h.size(0), -1, h.size(-1))  # reshape h to [B, N, C]
    
        h = torch.cat((cls_tokens, h), dim=1) # token: 1 + H + add_length
        n1 = h.shape[1] # n1 = 1 + H + add_length

        #---->Translayer x1
        h = self.layer1(h) #[B, N, dim_hid]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, dim_hid]
        
        #---->Translayer x2
        if 'ret_with_attn' in kwargs and kwargs['ret_with_attn']:
            h, attn = self.layer2(h, return_attn=True) # [B, N, dim_hid]
            # attn shape = [1, n_heads, n2, n2], where n2 = padding + n1
            if add_length == 0:
                attn = attn[:, :, -n1, (-n1+1):]
            else:
                attn = attn[:, :, -n1, (-n1+1):(-n1+1+H)]
            attn = attn.mean(1).detach()
            assert attn.shape[1] == H
        else:
            h = self.layer2(h) # [B, N, dim_hid]
            attn = None

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        #logits = torch.sigmoid(self._fc2(h))
        logits = self._fc2(h)
        
        if attn is not None:
            return logits, attn

        # Dummy values for saliency_maps, yhat_instance, and attention_scores to match ABMIL_aggregate's return type
        saliency_maps = None
        yhat_instance = None
        attention_scores = None

        return logits, saliency_maps, yhat_instance, attention_scores

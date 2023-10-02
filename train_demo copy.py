import timm, os
from timm import create_model
from fastai.vision.all import *
from fastai.vision.learner import _update_first_layer
import numpy as np
import torchvision.transforms as T
from torch import from_numpy
from torch import nn
import torch.utils.data as TUD
from nystrom_attention import NystromAttention


def recur_list(path):
    fname = []
    for root,d_names,f_names in os.walk(path):
        root_split = root.split('\\')
        subdir = os.path.join(root_split[-2],root_split[-1])
        for f in f_names:
            suffix = f.split('.')[-1]
            if suffix == 'JPEG':
                fname.append(os.path.join(subdir, f))
    return fname



def create_bags( files, labels, min_per_bag = 3, max_per_bag = 7, random_state = 42):
    '''
    input the list of (truncated) filenames and the label for each image
    output a nested list where each sublist is a list of filenames for the bag and bag_labels is a list of all the classes in the bag
    '''
    tot_instances = len(files)
    np.random.seed( seed = random_state )
    idx = np.random.permutation(tot_instances)
    
    bag_begin = 0
    rem_instances = tot_instances
    
    bag_files = []
    bag_labels = []
    bag_ids = []
    bag_sizes = []
    id = 0
    
    while rem_instances > min_per_bag:
        bag_size = np.random.randint( min_per_bag, max_per_bag + 1 )
        bag_idx = idx[ bag_begin: (bag_begin + bag_size) ]
        bag_files.append(files[ bag_idx ] )
        bag_labels.append(labels[ bag_idx ] )
        bag_ids.append(id*np.ones_like(bag_idx))
        id += 1
        bag_begin += bag_size
        rem_instances = tot_instances - bag_begin
        
    return bag_files, bag_labels, bag_ids

class BagOfImagesDataset(TUD.Dataset):

  def __init__(self, filenames, directory, ids, labels, imsize = 160, normalize=True):
    self.filenames = filenames
    self.directory = directory
    self.labels = from_numpy(labels)
    self.ids = from_numpy(ids)
    self.normalize = normalize
    self.imsize = imsize
  
    # Normalize
    if normalize:
        self.tsfms = T.Compose([
        T.ToTensor(),
        T.Resize( (self.imsize, self.imsize) ),
        T.Normalize(mean=[0.485, 0.456, 0.406],std= [0.229, 0.224, 0.225])
        ])
    else:
        self.tsfms = T.Compose([
        T.ToTensor(),
        T.Resize( (self.imsize, self.imsize) )
        ])

  def __len__(self):
    return len(torch.unique(self.ids))
  
  def __getitem__(self, index):
    where_id = self.ids == index
    files_this_bag = self.filenames[ where_id ]
    data = torch.stack( [ 
        self.tsfms( Image.open( os.path.join( self.directory, fn ) ).convert("RGB") ) for fn in files_this_bag 
    ] ).cuda()

    labels = self.labels[index]
    
    return data, labels
  
  def n_features(self):
    return self.data.size(1)

def collate_custom(batch):
    batch_data = []
    batch_bag_sizes = [0] 
    batch_labels = []
  
    for sample in batch:
        batch_data.append(sample[0])
        batch_bag_sizes.append(sample[0].shape[0])
        batch_labels.append(sample[1])
  
    out_data = torch.cat(batch_data, dim = 0).cuda()
    bagsizes = torch.IntTensor(batch_bag_sizes).cuda()
    out_bag_starts = torch.cumsum(bagsizes,dim=0).cuda()
    out_labels = torch.stack(batch_labels).cuda()
    
    return (out_data, out_bag_starts), out_labels



# this function is used to cut off the head of a pretrained timm model and return the body
def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or function")



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

        assert X.shape[0] == 1 # [1, n, 1024], single bag

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
        logits = torch.sigmoid(self._fc2(h))

        if attn is not None:
            return logits, attn

        return logits


class EmbeddingBagModel(nn.Module):
    
    def __init__(self, encoder, aggregator, num_classes=1):
        super(EmbeddingBagModel,self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.num_classes = num_classes
                    
    def forward(self, input):
        # input should be a tuple of the form (data, bag_starts)
        x = input[0]
        bag_sizes = input[1]
        
        # compute the features using encoder network
        h = self.encoder(x)
        
        # Here the shape of h is [B, C, H, W]. 
        # You need to change it to [B, N, C] as TransMIL expects it.
        # Assume H*W = N (total number of instances in a bag)
        h = h.view(h.size(0), -1, h.size(1))  # reshape h to [B, N, C]
        
        # TransMIL expects input of shape [1, N, C]
        # So, loop over the bags and compute logits for each
        num_bags = bag_sizes.shape[0]-1
        logits = torch.empty(num_bags, self.num_classes).cuda()
        
        for j in range(num_bags):
            start, end = bag_sizes[j], bag_sizes[j+1]
            h_bag = h[start:end]  # Extract instances for the current bag
            h_bag = h_bag.unsqueeze(0)  # Add a batch dimension
            logits[j] = self.aggregator(h_bag).squeeze(0)  # Remove the batch dimension from the output

        return logits  # The shape of logits is [num_bags, num_classes]


    
# "The regularization term |A| is basically model.saliency_maps.mean()" -from github repo
class L1RegCallback(Callback):
    def __init__(self, reglambda = 0.0001):
        self.reglambda = reglambda
       
    def after_loss(self):
        self.learn.loss += self.reglambda * self.learn.model.saliency_maps.mean()

def ilse_splitter(model):
    # split the model so that freeze works on the backbone
    p = params(model)
    num_body = len( params(model.encoder) )
    num_total = len(p)
    return [p[0:num_body], p[(num_body+1):num_total]]


# BCE loss doesn't encode the predictions, so 
# we include it in the accuracy
def accuracy_thresh(inp, targ, thresh=0.5):
    "Compute accuracy when `inp` and `targ` are the same size."
    inp,targ = flatten_check(inp,targ)
    return ((inp>thresh)==targ.bool()).float().mean()

path = untar_data(URLs.IMAGENETTE_160)
path_train = path/'train'
path_val = path/'val'


files_train = np.array(recur_list(path_train))
files_val = np.array(recur_list(path_val))


print(f'There are {len(files_train)} files in the training data')
print(f'There are {len(files_val)} files in the validation data')

lbl_dict = dict(
    n01440764='fish',
    n02102040='dog',
    n02979186='stereo',
    n03000684='saw',
    n03028079='church',
    n03394916='horn',
    n03417042='truck',
    n03425413='pump',
    n03445777='ball',
    n03888257='chute'
)

labels_train = np.array([ lbl_dict[ fn.split('\\')[1] ] for fn in files_train ])
labels_val = np.array([ lbl_dict[ fn.split('\\')[1] ] for fn in files_val ])


bags_val, bags_val_labels_all, bags_val_ids = create_bags( files_val, labels_val )
bags_train, bags_train_labels_all, bags_train_ids = create_bags( files_train, labels_train)

files_train = np.concatenate( bags_train )
ids_train = np.concatenate( bags_train_ids )
labels_train = np.array( [1 if 'fish' in x else 0 for x in bags_train_labels_all] )

files_val = np.concatenate( bags_val )
ids_val = np.concatenate( bags_val_ids )
labels_val = np.array( [1 if 'fish' in x else 0 for x in bags_val_labels_all] )


num_bags_train = 1882 # <= 1882
num_bags_val = 782 # <= 782

dataset_train = TUD.Subset(BagOfImagesDataset( files_train, path, ids_train, labels_train),list(range(0,num_bags_train)))
dataset_val = TUD.Subset(BagOfImagesDataset( files_val, path, ids_val, labels_val),list(range(0,num_bags_val)))
dataset_val_raw = TUD.Subset(BagOfImagesDataset( files_val, path, ids_val, labels_val, normalize = False),list(range(0,num_bags_val)))
bs = 10

train_dl =  TUD.DataLoader(dataset_train, batch_size=bs, collate_fn = collate_custom, drop_last=True, shuffle = True)
val_dl =    TUD.DataLoader(dataset_val, batch_size=bs, collate_fn = collate_custom, drop_last=True)

# wrap into fastai Dataloaders
dls = DataLoaders(train_dl, val_dl)

# set regularization parameter
reg_lambda = 0.001

encoder = create_timm_body('resnet18')
nf = num_features_model( nn.Sequential(*encoder.children()))

# bag aggregator
num_classes = 1
aggregator = TransMIL(dim_in=nf, dim_hid=512, n_classes=1)  # Adjust dim_hid and n_classes as needed

# total model
bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()

callbacks = [L1RegCallback(reg_lambda)]


learn = Learner(dls,bagmodel, loss_func=BCELossFlat(),metrics = accuracy_thresh )
learn.freeze()

# how many frozen / unfrozen parameters
frozen_parameters = filter(lambda p: not p.requires_grad, learn.model.parameters())
frozen = sum([np.prod(p.size()) for p in frozen_parameters])
unfrozen_parameters = filter(lambda p: p.requires_grad, learn.model.parameters())
unfrozen = sum([np.prod(p.size()) for p in unfrozen_parameters])
print(f'There are {frozen} frozen paraemters and {unfrozen} unfrozen parameters')

# find a good learning rate using mini-batches
learn.lr_find()

lr = 0.0008
learn.fit_one_cycle(10,lr)
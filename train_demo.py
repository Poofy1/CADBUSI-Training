import timm, os
from timm import create_model
from fastai.vision.all import *
from fastai.vision.learner import _update_first_layer
import numpy as np
import torchvision.transforms as T
from torch import from_numpy
from torch import nn
import torch.utils.data as TUD


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


class ABMIL_aggregate(nn.Module):
    
    def __init__(self, nf = 512, num_classes = 1, pool_patches = 3, L = 128):
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
    
    

class EmbeddingBagModel(nn.Module):
    
    def __init__(self, encoder, aggregator, num_classes = 1):
        super(EmbeddingBagModel,self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.num_classes = num_classes
                    
    def forward(self, input):
        # input should be a tuple of the form (data,bag_starts)
        x = input[0]
        bag_sizes = input[1]
        
        # compute the features using encoder network
        h = self.encoder(x)
        
        # loop over the bags and compute yhat_bag, etc. for each
        num_bags = bag_sizes.shape[0]-1
        saliency_maps, yhat_instances, attention_scores = [],[],[]
        
        yhat_bags = torch.empty(num_bags,self.num_classes).cuda()

        for j in range(num_bags):
            start, end = bag_sizes[j], bag_sizes[j+1]
            
            yhat_tmp, sm, yhat_ins, att_sc = self.aggregator(h[start:end])
            
            yhat_bags[j] = yhat_tmp
            saliency_maps.append(sm)
            yhat_instances.append(yhat_ins)
            attention_scores.append(att_sc)
            
        # converts lists to tensors (this seems optional)
        self.saliency_maps = torch.cat(saliency_maps,dim=0).cuda()
        self.yhat_instances = torch.cat(yhat_instances,dim=0).cuda()
        self.attention_scores = torch.cat(attention_scores,dim=0).cuda()
        
        # print(f'yhat_bags: {yhat_bags}')
        # print(f'Number of bags: {yhat_bags.shape}')
        # print(f'Saliency Maps: {self.saliency_maps.shape}')
        # print(f'yhat_instances: {self.yhat_instances}')
        # print(f'attention_scores: {self.attention_scores}')
       
        return yhat_bags
    
    
    
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

encoder = create_timm_body('resnet34')
nf = num_features_model( nn.Sequential(*encoder.children()))

# bag aggregator
num_classes = 1
aggregator = ABMIL_aggregate( nf = nf, num_classes = 1, pool_patches = 3, L = 128)

# total model
bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()

callbacks = [L1RegCallback(reg_lambda)]


learn = Learner(dls,bagmodel, loss_func=BCELossFlat(),splitter=ilse_splitter,metrics = accuracy_thresh, cbs = callbacks )
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
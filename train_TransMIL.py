import os
from timm import create_model
from fastai.vision.all import *
import torch.utils.data as TUD
from fastai.vision.learner import _update_first_layer
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
from torch import from_numpy
from torch import nn
from training_eval import *
from torch.optim import Adam
from data_prep import *
import torchvision.transforms.functional as TF
from nystrom_attention import NystromAttention
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True


class BagOfImagesDataset(TUD.Dataset):

    def __init__(self, filenames, ids, labels, normalize=True):
        self.filenames = filenames
        self.labels = from_numpy(labels)
        self.ids = from_numpy(ids)
        self.normalize = normalize
    
        # Normalize
        if normalize:
            self.tsfms = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAffine(
                    degrees=(-20, 20),  # Random rotation between -10 and 10 degrees
                    translate=(0.05, 0.05),  # Slight translation
                    scale=(0.95, 1.05),  # Slight scaling
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tsfms = T.Compose([
                T.ToTensor(),
            ])

    def __len__(self):
        return len(torch.unique(self.ids))
    
    def __getitem__(self, index):
        where_id = self.ids == index
        files_this_bag = self.filenames[where_id]
        data = torch.stack([
            self.tsfms(Image.open(fn).convert("RGB")) for fn in files_this_bag
        ]).cuda()

        labels = self.labels[index]

        return data, labels

    def show_image(self, index, img_index=0):
        # Get the transformed image tensor and label
        data, labels = self.__getitem__(index)

        # Select the specified image from the bag
        img_tensor = data[img_index]

        # If the images were normalized, reverse the normalization
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).to(img_tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(img_tensor.device)
            img_tensor = img_tensor * std[:, None, None] + mean[:, None, None]  # Unnormalize

        # Convert the image tensor to a PIL Image
        img = TF.to_pil_image(img_tensor.cpu())

        # Display the image and label
        plt.imshow(img)
        plt.title(f'Label: {labels}')
        plt.axis('off')  # Hide the axis
        plt.show()

    
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
        logits = self._fc2(h) #[B, n_classes]

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




if __name__ == '__main__':

    # Config
    model_name = 'ABMIL'
    img_size = 256
    batch_size = 4
    min_bag_size = 2
    max_bag_size = 15
    epochs = 15
    lr = 0.0008

    # Paths
    export_location = 'D:/DATA/CASBUSI/exports/export_09_28_2023/'
    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
    image_data = pd.read_csv(f'{export_location}/ImageData.csv')
    cropped_images = f"F:/Temp_SSD_Data/{img_size}_images/"
    
    
    
    files_train, ids_train, labels_train, files_val, ids_val, labels_val = prepare_all_data(export_location, case_study_data, breast_data, image_data, 
                                                                                            cropped_images, img_size, min_bag_size, max_bag_size)
    
    
    
    
    print("Training Data...")
    # Create datasets
    #dataset_train = TUD.Subset(BagOfImagesDataset( files_train, ids_train, labels_train),list(range(0,100)))
    #dataset_val = TUD.Subset(BagOfImagesDataset( files_val, ids_val, labels_val),list(range(0,100)))
    dataset_train = BagOfImagesDataset(files_train, ids_train, labels_train, img_size)
    dataset_val = BagOfImagesDataset(files_val, ids_val, labels_val, img_size)

        
    # Create data loaders
    train_dl =  TUD.DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)


    encoder = create_timm_body('resnet18')
    nf = num_features_model( nn.Sequential(*encoder.children()))
    
    # bag aggregator
    aggregator = TransMIL(dim_in=nf, dim_hid=512, n_classes=1)  # Adjust dim_hid and n_classes as needed

    # total model
    bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
        
        
    optimizer = Adam(bagmodel.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    all_targs = []
    all_preds = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        bagmodel.train()
        total_loss = 0.0
        total_acc = 0
        total = 0
        correct = 0
        for (data, yb) in tqdm(train_dl, total=len(train_dl)): 
            xb, ids = data
            xb, ids, yb = xb.cuda(), ids.cuda(), yb.cuda()
            optimizer.zero_grad()
            
            outputs = torch.sigmoid(bagmodel((xb, ids)).squeeze(dim=1))

            loss = loss_func(outputs, yb)
            
            #print(f'loss: {loss}\n pred: {outputs}\n true: {yb}')
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            predicted = torch.round(outputs).squeeze()
            total += yb.size(0)
            correct += predicted.eq(yb.squeeze()).sum().item() 
            
            if epoch == epochs - 1:
                all_targs.extend(yb.cpu().numpy())
                if len(predicted.size()) == 0:
                    predicted = predicted.view(1)
                all_preds.extend(predicted.cpu().detach().numpy())
            

        train_loss = total_loss / total
        train_acc = correct / total


        # Evaluation phase
        bagmodel.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for (data, yb) in tqdm(val_dl, total=len(val_dl)): 
                xb, ids = data  
                xb, ids, yb = xb.cuda(), ids.cuda(), yb.cuda()
                
                outputs = torch.sigmoid(bagmodel((xb, ids)).squeeze(dim=1))
                loss = loss_func(outputs, yb)
                
                total_val_loss += loss.item() * len(xb)
                predicted = torch.round(outputs).squeeze() 
                total += yb.size(0)
                correct += predicted.eq(yb.squeeze()).sum().item()

        val_loss = total_val_loss / total
        val_acc = correct / total
        
        train_losses_over_epochs.append(train_loss)
        valid_losses_over_epochs.append(val_loss)
        
        print(f"Epoch {epoch+1} | Acc   | Loss")
        print(f"Train   | {train_acc:.4f} | {train_loss:.4f}")
        print(f"Val     | {val_acc:.4f} | {val_loss:.4f}")
    
    # Save the model
    torch.save(bagmodel.state_dict(), f"{env}/models/{model_name}.pth")

    # Save the loss graph
    plot_loss(train_losses_over_epochs, valid_losses_over_epochs, f"{env}/models/{model_name}_loss.png")
    
    # Save the confusion matrix
    vocab = ['not malignant', 'malignant']  # Replace with your actual vocab
    plot_Confusion(all_targs, all_preds, vocab, f"{env}/models/{model_name}_confusion.png")

        
    
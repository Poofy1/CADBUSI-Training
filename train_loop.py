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
from model_ABMIL import *
from model_TransMIL import *
import torchvision.transforms.functional as TF
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True


class BagOfImagesDataset(TUD.Dataset):

    def __init__(self, filenames, ids, labels, train=True):
        self.filenames = filenames
        self.labels = from_numpy(labels)
        self.ids = from_numpy(ids)
        self.train = train
    
        # Normalize
        if train:
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
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(torch.unique(self.ids))
    
    def __getitem__(self, index):
        where_id = (self.ids == index).cpu().numpy()
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




class EmbeddingBagModel(nn.Module):
    
    def __init__(self, encoder, aggregator, num_classes=1):
        super(EmbeddingBagModel,self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.num_classes = num_classes
                    
                
    def forward(self, input):
        x = input[0]
        bag_sizes = input[1]
        h = self.encoder(x)
        h = h.view(h.size(0), -1, h.size(1))
        
        num_bags = bag_sizes.shape[0]-1
        logits = torch.empty(num_bags, self.num_classes).cuda()
        
        # Additional lists to store the saliency_maps, yhat_instances, and attention_scores
        saliency_maps, yhat_instances, attention_scores = [], [], []
        
        for j in range(num_bags):
            start, end = bag_sizes[j], bag_sizes[j+1]
            h_bag = h[start:end]
            
            # Ensure that h_bag has a first dimension of 1 before passing it to the aggregator
            h_bag = h_bag.unsqueeze(0)
            
            # Receive four values from aggregator
            yhat_bag, sm, yhat_ins, att_sc = self.aggregator(h_bag)
            
            logits[j] = yhat_bag
            saliency_maps.append(sm)
            yhat_instances.append(yhat_ins)
            attention_scores.append(att_sc)

        return logits


def BagMixUp(xb, ids, yb, alpha):
    # Pseudo-Bag Mixup
    lam = np.random.beta(alpha, alpha)

    # Random indices for the bags
    rand_index_bags = torch.randperm(yb.size()[0]).cuda()

    # Get the bag sizes from the ids tensor
    bag_sizes = [ids[i+1] - ids[i] for i in range(ids.shape[0] - 1)]
    mixed_bag_sizes = [bag_sizes[i] for i in rand_index_bags]

    # Create a new mixed_ids tensor
    mixed_ids = torch.tensor([0] + mixed_bag_sizes).cumsum(0).cuda()

    mixed_xb_list = []
    for i, rand_idx in enumerate(rand_index_bags):
        bag1 = xb[ids[i]:ids[i+1]]
        bag2 = xb[ids[rand_idx]:ids[rand_idx+1]]
        
        max_size = max(bag1.size(0), bag2.size(0))
        if bag1.size(0) < max_size:
            padding = torch.zeros((max_size - bag1.size(0),) + bag1.size()[1:]).cuda()
            bag1 = torch.cat([bag1, padding])
        if bag2.size(0) < max_size:
            padding = torch.zeros((max_size - bag2.size(0),) + bag2.size()[1:]).cuda()
            bag2 = torch.cat([bag2, padding])
        
        mixed_bag = lam * bag1 + (1-lam) * bag2
        mixed_xb_list.append(mixed_bag)

    mixed_xb = torch.cat(mixed_xb_list)

    # Mixed label based on contribution
    mixed_yb = yb if lam > 0.5 else yb[rand_index_bags]
    
    return mixed_xb, mixed_ids, mixed_yb
    


if __name__ == '__main__':

    # Config
    model_name = 'MixupTest2'
    img_size = 256
    batch_size = 6
    min_bag_size = 2
    max_bag_size = 15
    epochs = 150
    lr = 0.0008
    alpha = 0.4  # hyperparameter for the beta distribution

    # Paths
    #export_location = 'D:/DATA/CASBUSI/exports/export_09_28_2023/'
    #cropped_images = f"F:/Temp_SSD_Data/{img_size}_images/"
    export_location = '/home/hansen6528/DATA/export_09_28_2023/'
    cropped_images = f"/home/hansen6528/DATA/Temp_Data/{img_size}_images/"
    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
    image_data = pd.read_csv(f'{export_location}/ImageData.csv')
    
    
    
    
    files_train, ids_train, labels_train, files_val, ids_val, labels_val = prepare_all_data(export_location, case_study_data, breast_data, image_data, 
                                                                                            cropped_images, img_size, min_bag_size, max_bag_size)


    print("Training Data...")
    # Create datasets
    #dataset_train = TUD.Subset(BagOfImagesDataset( files_train, ids_train, labels_train),list(range(0,100)))
    #dataset_val = TUD.Subset(BagOfImagesDataset( files_val, ids_val, labels_val),list(range(0,100)))
    dataset_train = BagOfImagesDataset(files_train, ids_train, labels_train)
    dataset_val = BagOfImagesDataset(files_val, ids_val, labels_val, train=False)

        
    # Create data loaders
    train_dl =  TUD.DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)


    encoder = create_timm_body('resnet50')
    nf = num_features_model( nn.Sequential(*encoder.children()))
    
    # bag aggregator
    aggregator = ABMIL_aggregate( nf = nf, num_classes = 1, pool_patches = 3, L = 128)
    #aggregator = TransMIL(dim_in=nf, dim_hid=512, n_classes=1)

    # total model
    bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
    total_params = sum(p.numel() for p in bagmodel.parameters())
    print(f"Total Parameters: {total_params}")
        
        
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

            xb, ids, yb = BagMixUp(xb, ids, yb, alpha)

            optimizer.zero_grad()
            outputs = bagmodel((xb, ids)).squeeze(dim=1)
            loss = loss_func(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            predicted = torch.round(outputs).squeeze()
            total += yb.size(0)
            correct += predicted.eq(yb.squeeze()).sum().item()

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
                
                outputs = bagmodel((xb, ids)).squeeze(dim=1)
                loss = loss_func(outputs, yb)
                
                total_val_loss += loss.item() * len(xb)
                predicted = torch.round(outputs).squeeze() 
                total += yb.size(0)
                correct += predicted.eq(yb.squeeze()).sum().item()
                
                if epoch == epochs - 1:
                    all_targs.extend(yb.cpu().numpy())
                    if len(predicted.size()) == 0:
                        predicted = predicted.view(1)
                    all_preds.extend(predicted.cpu().detach().numpy())

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

        
    
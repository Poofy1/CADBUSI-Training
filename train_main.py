import os
from timm import create_model
from fastai.vision.all import *
from torch.utils.data import Dataset, Subset
from fastai.vision.learner import _update_first_layer
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
from torch import nn
from training_eval import *
from torch.optim import Adam
from data_prep import *
env = os.path.dirname(os.path.abspath(__file__))



class BagOfImagesDataset(Dataset):

    def __init__(self, data, imsize, normalize=True):
        self.bags = data
        self.normalize = normalize
        self.imsize = imsize

        # Normalize
        if normalize:
            self.tsfms = T.Compose([
                T.ToTensor(),
                #T.Resize((self.imsize, self.imsize)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tsfms = T.Compose([
                T.ToTensor(),
                #T.Resize((self.imsize, self.imsize))
            ])

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        filenames = bag[0]
        labels = bag[1]
        ids = bag[2]
        
        data = torch.stack([
            self.tsfms(Image.open(fn).convert("RGB")) for fn in filenames
        ]).cuda()
        
        # Convert the first label from the bag labels to a scalar tensor
        label = torch.tensor(labels[0], dtype=torch.float).cuda()
        
        # Convert bag ids to tensor with the same shape as the old data loader
        bagid = torch.full((len(filenames),), ids[0], dtype=torch.float).cuda()
        
        
        return data, bagid, label


def collate_custom(batch):
    batch_data = []
    batch_bagids = []
    batch_labels = []
  
    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])
  
    out_data = torch.cat(batch_data, dim = 0).cuda()
    out_bagids = torch.cat(batch_bagids).cuda()
    out_labels = torch.stack(batch_labels).cuda()
  
    return (out_data, out_bagids), out_labels



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



class IlseBagModel(nn.Module):
    
    def __init__(self, arch, num_classes = 1, pool_patches = 3, pretrained = True):
        super(IlseBagModel,self).__init__()
        self.pool_patches = pool_patches # how many patches to use in predicting instance label
        self.backbone = create_timm_body(arch, pretrained = pretrained)
        self.nf = num_features_model( nn.Sequential(*self.backbone.children()))
        self.num_classes = num_classes # two for binary classification
        self.M = self.nf # is 512 for resnet34
        self.L = 128 # number of latent features in gated attention     
        
        self.saliency_layer = nn.Sequential(        
            nn.Conv2d( self.nf, self.num_classes, (1,1), bias = False),
            nn.Sigmoid() )
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Sigmoid()
        )

        self.attention_W = nn.Linear(self.L, self.num_classes)
    
                    
    def forward(self, input):
        # input should be a tuple of the form (data,ids)
        ids = input[1]
        x = input[0]
        
        # compute the features using backbone network
        h = self.backbone(x)
        
        # add attention head to compute instance saliency map and instance labels (as logits)
    
        self.saliency_map = self.saliency_layer(h) # compute activation maps
        map_flatten = self.saliency_map.flatten(start_dim = -2, end_dim = -1)
        selected_area = map_flatten.topk(self.pool_patches, dim=2)[0]
        self.yhat_instance = selected_area.mean(dim=2).squeeze()
        
        # max pool the feature maps to generate feature vector v of length self.nf (number of features)
        v = torch.max( h, dim = 2).values
        v = torch.max( v, dim = 2).values # maxpool complete
        
        # gated-attention
        A_V = self.attention_V(v) 
        A_U = self.attention_U(v) 
        A  = self.attention_W(A_V * A_U).squeeze()
        
        unique = torch.unique_consecutive(ids)
        num_bags = len(unique)
        yhat_bags = torch.empty(num_bags,self.num_classes).cuda()
        for i,bag in enumerate(unique):
            mask = torch.where(ids == bag)[0]
            A[mask] = nn.functional.softmax( A[mask] , dim = 0 )
            yhat = self.yhat_instance[mask]

            yhat_bags[i] = ( A[mask] * yhat ).sum(dim=0)
        
        self.attn_scores = A
       
        return yhat_bags



if __name__ == '__main__':

    model_name = 'test1'
    img_size = 512
    batch_size = 3
    bag_size = 8
    epochs = 20   
    reg_lambda = 0.001
    lr = 0.001

    print("Preprocessing Data...")
    
    # Load CSV data
    export_location = 'F:/Temp_SSD_Data/export_09_14_2023/'
    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
    image_data = pd.read_csv(f'{export_location}/ImageData.csv')
    data = filter_raw_data(breast_data, image_data)

    #Cropping images
    cropped_images = f"{export_location}/temp_cropped/"
    preprocess_and_save_images(data, export_location, cropped_images, img_size)

    # Split the data into training and validation sets
    train_patient_ids = case_study_data[case_study_data['valid'] == 0]['Patient_ID']
    val_patient_ids = case_study_data[case_study_data['valid'] == 1]['Patient_ID']
    train_data = data[data['Patient_ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Patient_ID'].isin(val_patient_ids)].reset_index(drop=True)

    train_bags = create_bags(train_data, bag_size, cropped_images, inside_bag_upsampling=True)
    val_bags = create_bags(val_data, bag_size, cropped_images, inside_bag_upsampling=True) 
    
    print(f'There are {len(train_data)} files in the training data')
    print(f'There are {len(val_data)} files in the validation data')
    malignant_count, non_malignant_count = count_malignant_bags(train_bags)
    print(f"Number of Malignant Bags: {malignant_count}")
    print(f"Number of Non-Malignant Bags: {non_malignant_count}")
    
    
    
    print("Training Data...")

    train_amount = list(range(0,10))
    val_amount = list(range(0,10))
    
    # Create datasets
    #dataset_train = Subset(BagOfImagesDataset(train_bags, img_size), train_amount)
    #dataset_val = Subset(BagOfImagesDataset(val_bags, img_size), val_amount)
    dataset_train = BagOfImagesDataset(train_bags, img_size)
    dataset_val = BagOfImagesDataset(val_bags, img_size)
        
    # Create data loaders
    train_dl =  DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)

    bagmodel = IlseBagModel('resnet50', pretrained = True).cuda()
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
        for (xb, ids, yb) in tqdm(train_dl, total=len(train_dl)): 
            xb, ids, yb = xb.cuda(), ids.cuda(), yb.cuda()
            optimizer.zero_grad()
            
            xb = xb.view(-1, 3, img_size, img_size)
            outputs = bagmodel((xb, ids)).squeeze(dim=1)
            loss = loss_func(outputs, yb)
            
            #print(f'loss: {loss}\n pred: {outputs}\n true: {yb}')
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            predicted = torch.round(outputs).squeeze()
            total += yb.size(0)
            correct += predicted.eq(yb.squeeze()).sum().item() 
            
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
            for (xb, ids, yb) in tqdm(val_dl, total=len(val_dl)):   
                xb, ids, yb = xb.cuda(), ids.cuda(), yb.cuda()
                
                xb = xb.view(-1, 3, img_size, img_size)
                outputs = bagmodel((xb, ids)).squeeze(dim=1)
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

        
    
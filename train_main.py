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



def ilse_splitter(model):
    # split the model so that freeze works on the backbone
    p = params(model)
    num_body = len( params(model.encoder) )
    num_total = len(p)
    return [p[0:num_body], p[(num_body+1):num_total]]


if __name__ == '__main__':

    model_name = 'test1'
    img_size = 400
    batch_size = 4
    min_bag_size = 3
    max_bag_size = 10
    epochs = 20
    l1_lambda = 0 #0.001
    lr = 0.0008

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

    #data.to_csv(f'{env}/testData.csv')

    bags_train, bags_train_labels_all, bags_train_ids = create_bags(train_data, min_bag_size, max_bag_size, cropped_images)
    bags_val, bags_val_labels_all, bags_val_ids = create_bags(val_data, min_bag_size, max_bag_size, cropped_images)
    
    files_train = np.concatenate( bags_train )
    ids_train = np.concatenate( bags_train_ids )
    labels_train = np.array([1 if np.any(x == 1) else 0 for x in bags_train_labels_all], dtype=np.float32)

    files_val = np.concatenate( bags_val )
    ids_val = np.concatenate( bags_val_ids )
    labels_val = np.array([1 if np.any(x == 1) else 0 for x in bags_val_labels_all], dtype=np.float32)
    
    print(f'There are {len(files_train)} files in the training data')
    print(f'There are {len(files_val)} files in the validation data')
    malignant_count, non_malignant_count = count_bag_labels(labels_train)
    print(f"Number of Malignant Bags: {malignant_count}")
    print(f"Number of Non-Malignant Bags: {non_malignant_count}")
    
    
    
    print("Training Data...")
    # Create datasets
    dataset_train = BagOfImagesDataset(files_train, ids_train, labels_train, img_size)
    dataset_val = BagOfImagesDataset(files_val, ids_val, labels_val, img_size)

        
    # Create data loaders
    train_dl =  TUD.DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)


    encoder = create_timm_body('resnet50')
    nf = num_features_model( nn.Sequential(*encoder.children()))
    
    # bag aggregator
    aggregator = ABMIL_aggregate( nf = nf, num_classes = 1, pool_patches = 3, L = 128)

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
            
            outputs = bagmodel((xb, ids)).squeeze(dim=1)
            loss = loss_func(outputs, yb)
            
            # L1 regularization
            l1_reg = sum(param.abs().sum() for param in bagmodel.parameters())
            loss = loss + l1_lambda * l1_reg
            
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
            for (data, yb) in tqdm(val_dl, total=len(val_dl)): 
                xb, ids = data  
                xb, ids, yb = xb.cuda(), ids.cuda(), yb.cuda()
                
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

        
    
import os
from timm import create_model
from fastai.vision.all import *
from torch.utils.data import Dataset, Subset
from fastai.vision.learner import _update_first_layer
from PIL import Image
from torchvision import transforms
from torch import nn
from data_prep import *
env = os.path.dirname(os.path.abspath(__file__))




class CASBUSI_Dataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.patient_ids = {idx: pid for idx, pid in enumerate(data['Patient_ID'].unique())}
    
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        # Get the patient ID for this index
        patient_id = self.patient_ids[index]
        
        # Filter the data to get the rows corresponding to this patient ID
        same_patient_data = self.data[self.data['Patient_ID'] == patient_id].reset_index(drop=True)
        
        # Load and transform the images corresponding to this patient ID into a tensor
        images = []
        for i in range(len(same_patient_data)):
            img_name = os.path.join(self.root_dir, same_patient_data.iloc[i]['ImageName'])
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # Convert list of image tensors into a single 4D tensor
        images = torch.stack(images).cuda()
        
        # Get the label for this patient ID
        target = torch.tensor(same_patient_data.loc[0, ['Has_Malignant']].values.astype('int')).long().cuda()
        
        # Create a tensor for bag IDs, all having the same index value as the patient ID
        bagids = torch.full((len(same_patient_data),), index, dtype=torch.long).cuda()
        
        return images, bagids, target
    
    def n_features(self):
        return self.data.size(1)



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
    
    def __init__(self, arch, num_classes = 2, pool_patches = 3, pretrained = True):
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
    
                    
    def forward(self, x, ids):
        x = x.squeeze(0)
        
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
        A  = self.attention_W(A_V * A_U)
        
        unique = torch.unique_consecutive(ids)
        yhat_bags = torch.empty(len(unique),self.num_classes).cuda()
        for i,bag in enumerate(unique):
            mask = torch.where(ids == bag)[0]
            A[mask] = nn.functional.softmax( A[mask] , dim = 0 )
            yhat = self.yhat_instance[mask]
            yhat_bags[i] = ( A[mask] * yhat ).sum(dim=0)
        
        self.attn_scores = A
        return yhat_bags
    
# "The regularization term |A| is basically model.saliency_maps.mean()" -from github repo
class L1RegCallback(Callback):
    def __init__(self, reglambda = 0.0001):
        self.reglambda = reglambda
       
    def after_loss(self):
        self.learn.loss += self.reglambda * self.learn.model.saliency_map.mean()

if __name__ == '__main__':
    
    torch.cuda.empty_cache()

    img_size = 256
    batch_size = 1
    reg_lambda = 0.001
    lr = 0.0008
    
    
    # DATA PREP
    
    # Load CSV data
    export_location = 'F:/Temp_SSD_Data/export_09_14_2023/'
    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
    image_data = pd.read_csv(f'{export_location}/ImageData.csv')

    # Join dataframes on PatientID
    data = pd.merge(breast_data, image_data, left_on=['Patient_ID', 'Breast'], right_on=['Patient_ID', 'laterality'], suffixes=('', '_image_data'))

    # Remove columns from image_data that also exist in breast_data
    for col in breast_data.columns:
        if col + '_image_data' in data.columns:
            data.drop(col + '_image_data', axis=1, inplace=True)
    
    
    data = data[data['Has_Unknown'] == False]

    # Filter out patients with more than x data points
    patient_counts = data['Patient_ID'].value_counts()
    valid_patient_ids = patient_counts[patient_counts <= 20].index
    data = data[data['Patient_ID'].isin(valid_patient_ids)]
    
    # Group by 'Patient_ID' and apply the function to each group
    data = data.groupby('Patient_ID').apply(lambda group: upsample_bag_to_min_count(group, min_count=20))


    # Reset the index
    data.reset_index(drop=True, inplace=True)

    #Preparing data
    cropped_images = f"{export_location}/temp_cropped/"
    preprocess_and_save_images(data, export_location, cropped_images, img_size)

    # Split the data into training and validation sets
    train_patient_ids = case_study_data[case_study_data['valid'] == 0]['Patient_ID']
    val_patient_ids = case_study_data[case_study_data['valid'] == 1]['Patient_ID']
    train_data = data[data['Patient_ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Patient_ID'].isin(val_patient_ids)].reset_index(drop=True)



    print(f'There are {len(train_data)} files in the training data')
    print(f'There are {len(val_data)} files in the validation data')


    # Define transformations
    train_transform = transforms.Compose([
        GrayscaleToRGB(),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        GrayscaleToRGB(), 
        transforms.ToTensor()
    ])


    # Create datasets
    train_dataset = CASBUSI_Dataset(train_data, f'{cropped_images}/', transform=train_transform)
    val_dataset = CASBUSI_Dataset(val_data, f'{cropped_images}/', transform=val_transform)
        
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_custom)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_custom)

    # wrap into fastai Dataloaders
    dls = DataLoaders(train_loader, val_loader)


    timm_arch = 'resnet18' #resnet34
    bagmodel = IlseBagModel(timm_arch, pretrained = True).cuda()
    
    learn = Learner(dls, bagmodel, loss_func=CrossEntropyLossFlat(), metrics = accuracy, cbs = L1RegCallback(reg_lambda) )

    # find a good learning rate using mini-batches
    learn.lr_find()
    
    learn.fit_one_cycle(10,lr)
    
    learn.save(f"{env}/models/test1")
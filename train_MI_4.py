import random, os
from timm import create_model
from fastai.vision.all import *
from torch.utils.data import Dataset
from fastai.vision.learner import _update_first_layer
from PIL import Image
import numpy as np
from torchvision import transforms
from torch import nn
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
env = os.path.dirname(os.path.abspath(__file__))


def process_single_image(row, root_dir, output_dir, resize_and_pad):
    patient_id = row['Patient_ID']
    img_name = row['ImageName']
    input_path = os.path.join(f'{root_dir}images/', img_name)
    output_path = os.path.join(output_dir, img_name)

    if os.path.exists(output_path):  # Skip images that are already processed
        return

    image = Image.open(input_path)
    image = resize_and_pad(image)
    image.save(output_path)

def preprocess_and_save_images(data, root_dir, output_dir, image_size, fill=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Preprocessing Data")

    resize_and_pad = ResizeAndPad(image_size, fill)
    data_rows = [row for _, row in data.iterrows()]  # Convert the DataFrame to a list of rows

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, row, root_dir, output_dir, resize_and_pad): row for row in data_rows}

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                future.result()  # We don't actually use the result, but this will raise any exceptions
                pbar.update()

class GrayscaleToRGB:
    def __call__(self, img):
        if len(img.getbands()) == 1:  # If image is grayscale
            img = transforms.functional.to_pil_image(np.stack([img] * 3, axis=-1))
        return img

class ResizeAndPad:
    def __init__(self, output_size, fill=0):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if h > w:
            new_h, new_w = self.output_size, int(self.output_size * (w / h))
        else:
            new_h, new_w = int(self.output_size * (h / w)), self.output_size
        img = transforms.functional.resize(img, (new_h, new_w))

        diff = self.output_size - new_w if h > w else self.output_size - new_h
        padding = [diff // 2, diff // 2]

        # If the difference is odd, add the extra padding to the end
        if diff % 2 != 0:
            padding[1] += 1

        # Use the padding values for the left/right or top/bottom
        padding = (padding[0], 0, padding[1], 0) if h > w else (0, padding[0], 0, padding[1])
        img = transforms.functional.pad(img, padding, fill=self.fill)
        return img


class CASBUSI_Dataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.patient_ids = {idx: pid for idx, pid in enumerate(data['Patient_ID'].unique())}

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        same_patient_data = self.data[self.data['Patient_ID'] == patient_id].reset_index(drop=True)
        
        # Now that we've pre-filtered, we assume bag_size <= 40.
        bag_size = len(same_patient_data)
        
        images = []
        for i in range(bag_size):
            img_name = os.path.join(self.root_dir, same_patient_data.iloc[i]['ImageName'])
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)
        target = torch.tensor(same_patient_data.loc[0, ['Has_Malignant']].values.astype('float'))

        return (images, target), patient_id

def custom_collate(batch):
    return batch

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
    
                    
    def forward(self, input):
        # input should be a tuple of the form (data,ids)
        ids = input[1]
        x = input[0]
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        
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

def upsample_bag_to_min_count(group, min_count=10):
    num_needed = min_count - len(group)
    
    if num_needed > 0:
        # Duplicate random samples within the same group
        random_rows = group.sample(num_needed, replace=True)
        group = pd.concat([group, random_rows], axis=0).reset_index(drop=True)
        
    return group


if __name__ == '__main__':

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
    valid_patient_ids = patient_counts[patient_counts <= 10].index
    data = data[data['Patient_ID'].isin(valid_patient_ids)]
    
    # Group by 'Patient_ID' and apply the function to each group
    data = data.groupby('Patient_ID').apply(upsample_bag_to_min_count)

    # Reset the index
    data.reset_index(drop=True, inplace=True)

    #Preparing data
    cropped_images = f"{export_location}/temp_cropped/"
    #preprocess_and_save_images(data, export_location, cropped_images, img_size)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, collate_fn=custom_collate)

    # wrap into fastai Dataloaders
    dls = DataLoaders(train_loader, val_loader)


    timm_arch = 'resnet18' # 96-97%
    #timm_arch = 'resnet34' # gets about 98% accuracy at the bag-level
    #timm_arch = 'resnet152' # about 98%

    bagmodel = IlseBagModel(timm_arch,pretrained = True).cuda()

    
    learn = Learner(dls,bagmodel, loss_func=CrossEntropyLossFlat(), metrics = accuracy, cbs = L1RegCallback(reg_lambda) )


    # find a good learning rate using mini-batches
    learn.lr_find()

    
    learn.fit_one_cycle(10,lr)
    
    
    learn.save(f"{env}/models/test1")
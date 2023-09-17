import timm, torch, os
import torch.utils.data as TUD
import pandas as pd
import torchvision.transforms as T
from torch import from_numpy, nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")




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
    bagids = self.ids[where_id]
    labels = self.labels[index]

    return data, bagids, labels
  
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



def create_timm_body(arch, pretrained=True, n_in=3):
    model = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    # _update_first_layer functionality can be implemented here if required.
    
    # Finding the layer to cut at
    cut = None
    for i, layer in enumerate(list(model.children())):
        if isinstance(layer, torch.nn.AdaptiveAvgPool2d):
            cut = i
            break

    if cut is not None:
        return torch.nn.Sequential(*list(model.children())[:cut])
    else:
        raise ValueError("Could not find a layer to cut at.")
    
def num_features_model(model):
    dummy_input = torch.ones(1, 3, 160, 160)
    dummy_output = model(dummy_input)
    return dummy_output.size(1)
    
class IlseBagModel(nn.Module):
    
    def __init__(self, arch, num_classes = 2, pool_patches = 3, pretrained = True):
        super(IlseBagModel,self).__init__()
        self.pool_patches = pool_patches # how many patches to use in predicting instance label
        self.backbone = create_timm_body(arch, pretrained = pretrained)
        self.nf = num_features_model(self.backbone)
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
        return len(self.patient_ids)  # Return number of unique patients (bags)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        same_patient_data = self.data[self.data['Patient_ID'] == patient_id].reset_index(drop=True)
        bag_size = len(same_patient_data)
        
        if bag_size > 50:
            return None, None
        
        images = []
        for i in range(bag_size):
            img_name = os.path.join(self.root_dir, same_patient_data.iloc[i]['ImageName'])
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)

        has_unknown = same_patient_data.loc[0, 'Has_Unknown']
        if not has_unknown:
            #target = torch.tensor(same_patient_data.loc[0, ['Has_Malignant', 'Has_Benign']].values.astype('float'))
            target = torch.tensor(same_patient_data.loc[0, ['Has_Malignant']].values.astype('float'))
        else:
            target = None

        return images, target

def load_model(model, model_path):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print("No previous model found, starting training from scratch.")
    return model





if __name__ == "__main__":
    
    # Load CSV data
    export_location = 'F:/Temp_SSD_Data/export_09_14_2023/'
    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
    image_data = pd.read_csv(f'{export_location}/ImageData.csv')

    # Join dataframes on PatientID
    data = pd.merge(breast_data, image_data, left_on=['Patient_ID', 'Breast'], right_on=['Patient_ID', 'laterality'], suffixes=('', '_image_data'))

    
    model_name = 'model_09_02_2023'
    epochs = 2
    image_size = 512
    batch_size = 5
    
    output_size = 1
    


    
    #Preparing data
    cropped_images = f"{export_location}/temp_cropped/"
    preprocess_and_save_images(data, export_location, cropped_images, image_size)
        
    
    
    # Define transformations
    train_transform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])

    val_transform = transforms.Compose([
        GrayscaleToRGB(), 
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])

    # Split the data into training and validation sets
    train_patient_ids = case_study_data[case_study_data['valid'] == 0]['Patient_ID']
    val_patient_ids = case_study_data[case_study_data['valid'] == 1]['Patient_ID']
    train_data = data[data['Patient_ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Patient_ID'].isin(val_patient_ids)].reset_index(drop=True)

    # Create datasets
    train_dataset = CASBUSI_Dataset(train_data, f'{cropped_images}/', transform=train_transform)
    val_dataset = CASBUSI_Dataset(val_data, f'{cropped_images}/', transform=val_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    
    os.makedirs(f"{env}/models/", exist_ok=True)
    model = load_model(model, f"{env}/models/{model_name}.pth")
    #summary(model, input_size=(1, 1, image_size, image_size))
    print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

    train_model(model, model_name, epochs)

    arch = "resnet18" # replace with your choice of architecture
    model_body = create_timm_body(arch)
    print(model_body)
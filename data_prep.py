import os
from PIL import Image
from torchvision import transforms
from fastai.vision.all import *
import torch.utils.data as TUD
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
import cv2
import ast
from PIL import ImageOps
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.utils import resample
env = os.path.dirname(os.path.abspath(__file__))


class GaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()).mul_(self.std).add_(self.mean)
        return tensor.add_(noise)

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

        # Determine the longer side and calculate 5% of its length
        trim_percent = 0.05
        if h > w:
            trim_size = int(h * trim_percent)
            # Trim 5% from the top and bottom
            img = img.crop((0, trim_size, w, h - trim_size))
        else:
            trim_size = int(w * trim_percent)
            # Trim 5% from the left and right
            img = img.crop((trim_size, 0, w - trim_size, h))

        # Update new image size
        w, h = img.size

        # Resize the image
        if h > w:
            new_h, new_w = self.output_size, int(self.output_size * (w / h))
        else:
            new_h, new_w = int(self.output_size * (h / w)), self.output_size
        img = img.resize((new_w, new_h))

        # Calculate padding
        diff = self.output_size - new_w if h > w else self.output_size - new_h
        padding = [diff // 2, diff // 2]

        # If the difference is odd, add the extra padding to the end
        if diff % 2 != 0:
            padding[1] += 1

        # Apply padding
        padding = (padding[0], 0, padding[1], 0) if h > w else (0, padding[0], 0, padding[1])
        img = ImageOps.expand(img, border=padding, fill=self.fill)

        return img
    
    
    
class ResizeAndStretch:
    def __init__(self, output_size, fill=0):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        # Determine which dimension (width or height) is smaller
        if w < h:  # Width is smaller
            stretched_w = self.output_size
            stretched_h = int(h * (stretched_w / w))
        else:  # Height is smaller
            stretched_h = self.output_size
            stretched_w = int(w * (stretched_h / h))

        # Stretch the image
        img = img.resize((stretched_w, stretched_h), Image.ANTIALIAS)

        # Final resize to ensure the image is output_size x output_size
        img = img.resize((self.output_size, self.output_size), Image.ANTIALIAS)

        return img
    
class HistogramEqualization(object):
    def __call__(self, image):
        # Must be a PIL Image
        image_np = np.array(image)

        # Check if the image is grayscale or color and apply histogram equalization accordingly
        if len(image_np.shape) == 2:
            # Grayscale image
            image_eq = cv2.equalizeHist(image_np)
        else:
            # Color image
            image_eq = np.zeros_like(image_np)
            for i in range(image_np.shape[2]):
                image_eq[..., i] = cv2.equalizeHist(image_np[..., i])

        # Convert back to PIL Image
        image_eq = Image.fromarray(image_eq)
        return image_eq
    
def unnormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(tensor.device)
        tensor = tensor * std + mean  # unnormalize
        return torch.clamp(tensor, 0, 1)


class CLAHETransform(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img = np.array(img)
        if len(img.shape) == 2:
            img = self.clahe.apply(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            img[:, :, 0] = self.clahe.apply(img[:, :, 0])
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

class BagOfImagesDataset(TUD.Dataset):

    def __init__(self, bags_dict, train=True, save_processed=False):
        self.bags_dict = bags_dict
        self.unique_bag_ids = list(bags_dict.keys())
        self.save_processed = save_processed
        self.train = train
    
        # Normalize
        if train:
            self.tsfms = T.Compose([
                T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.05), scale=(1, 1.2),),
                #HistogramEqualization(),
                CLAHETransform(),
                T.ToTensor(),
                GaussianNoise(mean=0, std=0.015),  # Add slight noise
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tsfms = T.Compose([
                #HistogramEqualization(),
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, index):
        actual_id = self.unique_bag_ids[index]
        labels, files_this_bag = self.bags_dict[actual_id]
        data = torch.stack([
            self.tsfms(Image.open(fn).convert("RGB")) for fn in files_this_bag
        ])
        data = data.cuda() 
        
        if self.save_processed:
            save_folder = os.path.join(env, 'processed_images')  
            os.makedirs(save_folder, exist_ok=True)
            for idx, img_tensor in enumerate(data):
                img_save_path = os.path.join(save_folder, f'bag_{actual_id}_img_{idx}.png')
                img_tensor = unnormalize(img_tensor)
                img = TF.to_pil_image(img_tensor.cpu().detach())
                img.save(img_save_path)
                
        # Convert labels list to a tensor
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return data, label_tensor, actual_id
    
    def __len__(self):
        return len(self.unique_bag_ids)
    
    def n_features(self):
        return self.data.size(1)



def create_bags(data, min_size, max_size, root_dir, label_columns):
    bags_dict = {}  # This will be indexed by ID
    
    total_rows = len(data)

    for _, row in tqdm(data.iterrows(), total = total_rows):
        # Parse the 'Images' column to get the list of images
        image_files = ast.literal_eval(row['Images'])

        # Exclude bags that are outside the size range
        if not (min_size <= len(image_files) <= max_size):
            continue

        # Create the full path for each image file
        bag_files = [os.path.join(root_dir, img_name) for img_name in image_files]

        # Extract labels from the specified columns
        bag_labels = [int(row[label]) for label in label_columns]

        # Add to dictionary
        bags_dict[row['ID']] = [bag_labels, bag_files]

    return bags_dict  # ID : [[labels], [image_files]]



def count_bag_labels(bags_dict):
    label_combinations_count = {}

    for bag_id, bag_contents in bags_dict.items():
        # Convert the label list to a tuple so it can be used as a dictionary key
        label_tuple = tuple(bag_contents[0])

        if label_tuple not in label_combinations_count:
            label_combinations_count[label_tuple] = 0
        label_combinations_count[label_tuple] += 1

    # Print counts for each label combination
    for label_combination, count in label_combinations_count.items():
        print(f"Label combination {label_combination}: {count} bags")


def process_single_image(img_name, root_dir, output_dir, resize_and_pad):
    try:
        input_path = os.path.join(f'{root_dir}images/', img_name)
        output_path = os.path.join(output_dir, img_name)

        if os.path.exists(output_path):  # Skip images that are already processed
            return

        image = Image.open(input_path)
        image = resize_and_pad(image)
        image.save(output_path)
    except Exception as e:
        print(f"Error processing image {img_name}: {e}")

def preprocess_and_save_images(data, root_dir, output_dir, image_size, fill=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    resize_and_pad = ResizeAndPad(image_size, fill=fill)

    # Flatten the data to have a list of image names with their corresponding row data
    flattened_data = [(img_name, row) for _, row in data.iterrows() for img_name in ast.literal_eval(row['Images'])]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, img_name, root_dir, output_dir, resize_and_pad): (img_name, row) for img_name, row in flattened_data}

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions
                pbar.update()


def upsample_minority_class(bags_dict, seed=0):
    np.random.seed(seed)  # for reproducibility

    # Convert dict to list of tuples for easy processing
    bags_list = [(bag_id, info) for bag_id, info in bags_dict.items()]

    # Extract the first label from the label list for each bag
    class_0_bags = [bag for bag in bags_list if bag[1][0][0] == 0]
    class_1_bags = [bag for bag in bags_list if bag[1][0][0] == 1]

    # Identify the minority class
    minority_bags = class_0_bags if len(class_0_bags) < len(class_1_bags) else class_1_bags
    majority_bags = class_1_bags if len(class_0_bags) < len(class_1_bags) else class_0_bags

    # Calculate how many duplicates are needed
    num_to_oversample = len(majority_bags) - len(minority_bags)

    # Randomly duplicate minority class bags
    oversampled_bags = resample(minority_bags, n_samples=num_to_oversample, random_state=seed)
    max_existing_id = max(bags_dict.keys())

    # Add the oversampled bags to the bags_dict with new IDs
    for bag in oversampled_bags:
        max_existing_id += 1  # increment to get a new unique ID
        bags_dict[max_existing_id] = bag[1]

    return bags_dict
    

def save_bags_to_csv(bags_dict, file_path):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['Bag_ID', 'Labels', 'Image_Files'])

        for bag_id, bag_contents in bags_dict.items():
            # Flatten image file paths into a single string
            images_str = ';'.join(bag_contents[1])
            # Convert labels list to string
            labels_str = ','.join(map(str, bag_contents[0]))
            # Write row
            writer.writerow([bag_id, labels_str, images_str])

def prepare_all_data(export_location, label_columns, cropped_images, img_size, min_bag_size, max_bag_size):
    
    print("Preprocessing Data...")
    data = pd.read_csv(f'{export_location}/TrainData.csv')
       
    #Cropping images
    preprocess_and_save_images(data, export_location, cropped_images, img_size)
    
    # Split the data into training and validation sets
    train_patient_ids = data[data['Valid'] == 0]['ID']
    val_patient_ids = data[data['Valid'] == 1]['ID']
    train_data = data[data['ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['ID'].isin(val_patient_ids)].reset_index(drop=True)
    
    bags_train = create_bags(train_data, min_bag_size, max_bag_size, cropped_images, label_columns)
    bags_val = create_bags(val_data, min_bag_size, max_bag_size, cropped_images, label_columns)
    
    bags_train = upsample_minority_class(bags_train)  # Upsample the minority class in the training set
    
    print(f'There are {len(bags_train)} files in the training data')
    print(f'There are {len(bags_val)} files in the validation data')
    count_bag_labels(bags_train)
    save_bags_to_csv(bags_train, 'F:/Temp_SSD_Data/bags_train.csv')
    return bags_train, bags_val
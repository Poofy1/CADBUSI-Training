import os
import torch.utils.data as TUD
from fastai.vision.all import *
from tqdm import tqdm
import numpy as np
import ast
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data.distributed import DistributedSampler
from sklearn.utils import resample
from data.transforms import *
from storage_adapter import *
from data.bag_loader import *
from config import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)


def create_bags(config, data, root_dir, instance_data=None):
    label_columns = config['label_columns']
    instance_columns = config['instance_columns']
    min_size = config['min_bag_size']
    max_size = config['max_bag_size']
    use_videos = config.get('use_videos', False)
    
    bags_dict = {}
    image_label_map = {}
    
    # Process instance data if provided
    if instance_data is not None and instance_columns is not None:
        if isinstance(instance_data, pd.DataFrame):
            for _, row in instance_data.iterrows():
                image_name = row['ImageName']
                labels = []
                for col in instance_columns:
                    if col in instance_data.columns:
                        label_value = row[col]
                        labels.append(int(label_value) if isinstance(label_value, bool) else label_value)
                image_label_map[image_name] = labels
    
    total_rows = len(data)
    all_files = list_files(root_dir)
    
    # Create video prefix mapping
    video_prefix_map = {}
    if use_videos and 'VideoPaths' in data.columns:
        for f in all_files:
            basename = os.path.basename(f)
            prefix = '_'.join(basename.split('_')[:-1])
            if prefix not in video_prefix_map:
                video_prefix_map[prefix] = []
            video_prefix_map[prefix].append(f)

        for prefix in video_prefix_map:
            video_prefix_map[prefix].sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for _, row in tqdm(data.iterrows(), total=total_rows):
        image_files = ast.literal_eval(row['Images'])
        
        bag_files = []
        image_labels = []
        video_frames = []
        
        # First, collect all available video frames
        if use_videos and 'VideoPaths' in data.columns:
            video_prefixes = ast.literal_eval(row['VideoPaths'])
            for video_prefix in video_prefixes:
                if video_prefix in video_prefix_map:
                    video_frames.extend(video_prefix_map[video_prefix])
        
        # Process regular images
        video_filenames = set(os.path.basename(f) for f in video_frames)
        for img_name in image_files:
            labels = image_label_map.get(img_name, [None] * len(instance_columns)) if instance_columns else []
            
            if not use_videos or os.path.basename(img_name) not in video_filenames:
                full_path = os.path.join(root_dir, img_name)
                bag_files.append(full_path)
                image_labels.append(labels if any(label is not None for label in labels) else [None])
        
        # If we have too many images, skip this bag
        if len(bag_files) > max_size:
            continue
        
        # If we have video frames and aren't at max size yet, add some video frames (up to max_size)
        if use_videos and len(bag_files) < max_size and video_frames:
            frames_needed = max_size - len(bag_files)
            video_frames = video_frames[:frames_needed]
        else: 
            video_frames = []
            
        # Skip if we don't meet minimum size requirement (even with videos)
        if len(bag_files) + len(video_frames) < min_size:
            continue
        
        bag_labels = [int(row[label]) for label in label_columns if label in data.columns]
    
        bags_dict[row['ID']] = {
            'bag_labels': bag_labels, 
            'images': bag_files,
            'image_labels': image_labels,
            'videos': video_frames,
            'Accession_Number': row['Accession_Number']
        }

    return bags_dict  # ID : {'bag_labels': [...], 'images': [...], 'image_labels': [...], 'videos': [...], 'Accession_Number': xxx}




def count_bag_labels(bags_dict):
    label_combinations_count = {}

    for bag_id, bag_contents in bags_dict.items():
        # Access the 'bag_labels' key of the dictionary
        label_tuple = tuple(bag_contents['bag_labels'])

        if label_tuple not in label_combinations_count:
            label_combinations_count[label_tuple] = 0
        label_combinations_count[label_tuple] += 1

    # Print counts for each label combination
    for label_combination, count in label_combinations_count.items():
        print(f"Label combination {label_combination}: {count} bags")


def process_single_image(img_path, root_dir, output_dir, resize_and_pad, video_name = None):
    try:
        if video_name:
            input_path = os.path.join(root_dir, 'videos', img_path)
        else:
            input_path = os.path.join(root_dir, 'images', img_path)
        
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        
        if file_exists(output_path):
            return

        image = read_image(input_path, use_pil=True)
        if image is None:
            raise ValueError(f"Failed to read image: {input_path}")
            
        image = resize_and_pad(image)
        save_data(image, output_path)
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")

def preprocess_and_save_images(config, data, root_dir, output_dir, fill=0):
    make_dirs(output_dir, local_override = False)

    resize_and_pad = ResizeAndPad(config['img_size'], fill=fill)

    # Process regular images
    regular_images = [(img_name, False) for _, row in data.iterrows() 
                     for img_name in ast.literal_eval(row['Images'])]
    
    # Process video images using CSV
    video_images = []
    if config['use_videos']:
        video_data = read_csv(f'{root_dir}/VideoImages.csv')
        if video_data is not None:
            for _, row in video_data.iterrows():
                video_folder = row['video_name'].replace('\\', '/')  # Convert backslashes here
                image_paths = ast.literal_eval(row['images'])
                video_images.extend([
                    (os.path.basename(img_path).replace('\\', '/'), video_folder) 
                    for img_path in image_paths
                ])
            
    # Combine both image lists
    all_images = regular_images + video_images
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(
                process_single_image,
                img_path,
                root_dir,
                output_dir,
                resize_and_pad,
                video_name
            ): img_path 
            for img_path, video_name in all_images
        }
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update()


def upsample_minority_class(bags_dict, seed=0):
    np.random.seed(seed)  # for reproducibility

    # Convert dict to list of tuples for easy processing
    bags_list = [(bag_id, info) for bag_id, info in bags_dict.items()]

    # Extract the first label from the label list for each bag
    class_0_bags = [bag for bag in bags_list if bag[1]['bag_labels'][0] == 0]
    class_1_bags = [bag for bag in bags_list if bag[1]['bag_labels'][0] == 1]


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
        writer.writerow(['Bag_ID', 'Labels', 'Image_Files', 'Image_Labels'])

        for bag_id, bag_contents in bags_dict.items():
            # Flatten image file paths into a single string
            images_str = ';'.join(bag_contents['images'])
            # Convert bag labels list to string
            labels_str = ','.join(map(str, bag_contents['bag_labels']))
            # Serialize image-level labels to a JSON string
            image_labels_str = json.dumps(bag_contents['image_labels'])
            # Write row
            writer.writerow([bag_id, labels_str, images_str, image_labels_str])


def prepare_all_data(config):

    # Path to the config file
    export_location = f"{config['export_location']}/{config['dataset_name']}"
    cropped_images = f"{config['cropped_images']}/{config['dataset_name']}_{config['img_size']}_images"
    
    print("Preprocessing Data...")
    data = read_csv(f'{export_location}/TrainData.csv')

    instance_data_file = f'{export_location}/InstanceData.csv'
    
    if file_exists(instance_data_file):
        instance_data = read_csv(instance_data_file)
    else:
        instance_data = None
       
    #Cropping images
    preprocess_and_save_images(config, data, export_location, cropped_images)
    
    # Split the data into training and validation sets
    train_patient_ids = data[data['Valid'] == 0]['Accession_Number']
    val_patient_ids = data[data['Valid'] == 1]['Accession_Number']
    train_data = data[data['Accession_Number'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Accession_Number'].isin(val_patient_ids)].reset_index(drop=True)
    
    bags_train = create_bags(config, train_data, cropped_images, instance_data)
    bags_val = create_bags(config, val_data, cropped_images, instance_data)
    
    #bags_train = upsample_minority_class(bags_train)  # Upsample the minority class in the training set
    
    print(f'There are {len(bags_train)} files in the training data')
    print(f'There are {len(bags_val)} files in the validation data')
    count_bag_labels(bags_train)
    
    
    # Debug
    #save_bags_to_csv(bags_train, 'F:/Temp_SSD_Data/bags_testing.csv')
    
    
    
    # Create bag datasets
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
    bag_dataset_val = BagOfImagesDataset(bags_val, transform=val_transform)
    train_sampler = BalancedBagSampler(bag_dataset_train, batch_size=config['bag_batch_size'])
    val_sampler = BalancedBagSampler(bag_dataset_val, batch_size=config['bag_batch_size'])
    #train_sampler = DistributedBalancedBagSampler(bag_dataset_train, config['bag_batch_size'])
    #val_sampler = DistributedBalancedBagSampler(bag_dataset_val, config['bag_batch_size'])
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_sampler=train_sampler, collate_fn=collate_bag, num_workers=8, pin_memory=True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_sampler=val_sampler, collate_fn=collate_bag)
    

    return bags_train, bags_val, bag_dataloader_train, bag_dataloader_val
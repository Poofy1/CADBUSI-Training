import os
from PIL import Image
from fastai.vision.all import *
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import ast
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.utils import resample
from data.transforms import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

from config import load_paths

    



def create_bags(data, min_size, max_size, root_dir, label_columns, instance_columns=None, instance_data=None):
    bags_dict = {}  # Indexed by ID
    
    image_label_map = {}
    
    # Check if instance_data and instance_columns are provided and valid
    if instance_data is not None and instance_columns is not None:
        # Process instance_data only if it's a valid DataFrame
        if isinstance(instance_data, pd.DataFrame):
            for _, row in instance_data.iterrows():
                image_name = row['ImageName']
                # Process labels, translating booleans to integers
                labels = []
                for col in instance_columns:
                    if col in instance_data.columns:
                        label_value = row[col]
                        if isinstance(label_value, bool):
                            labels.append(int(label_value))  # Convert boolean to int
                        else:
                            labels.append(label_value)  # Keep as is for non-boolean values
                
                # Store the labels
                image_label_map[image_name] = labels
    
    total_rows = len(data)
    
    for _, row in tqdm(data.iterrows(), total=total_rows):
        image_files = ast.literal_eval(row['Images'])
    
        bag_files = []
        image_labels = []
    
        for img_name in image_files:
            # Get labels from image_label_map or use default labels if not available
            labels = image_label_map.get(img_name, [None] * len(instance_columns)) if instance_columns else []
    
            full_path = os.path.join(root_dir, img_name)
            bag_files.append(full_path)
    
            # Append [None] or the labels based on content
            image_labels.append(labels if any(label is not None for label in labels) else [None])

        # Skip bags outside the size range
        if not (min_size <= len(bag_files) <= max_size):
            continue
    
        # Extract labels from data
        bag_labels = [int(row[label]) for label in label_columns if label in data.columns]
    
        bags_dict[row['ID']] = {
            'bag_labels': bag_labels, 
            'images': bag_files,
            'image_labels': image_labels
        }

    return bags_dict  # ID : {'bag_labels': [...], 'images': [...], 'image_labels': [...]}




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


def process_single_image(img_name, root_dir, output_dir, resize_and_pad):
    try:
        input_path = os.path.join(f'{root_dir}/images/', img_name)
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
    
    label_columns = config['label_columns']
    instance_columns = config['instance_columns']
    img_size = config['img_size']
    min_bag_size = config['min_bag_size']
    max_bag_size = config['max_bag_size']
    
    # Path to the config file
    json_config = load_paths()
    export_location = f"{json_config['export_location']}/{config['dataset_name']}"
    cropped_images = f"{json_config['cropped_images']}/{config['dataset_name']}_{config['img_size']}_images"
    
    print("Preprocessing Data...")
    data = pd.read_csv(f'{export_location}/TrainData.csv')
    
    instance_data_file = f'{export_location}/InstanceData.csv'
    
    if os.path.exists(instance_data_file):
        instance_data = pd.read_csv(instance_data_file)
    else:
        instance_data = None
       
    #Cropping images
    preprocess_and_save_images(data, export_location, cropped_images, img_size)
    
    # Split the data into training and validation sets
    train_patient_ids = data[data['Valid'] == 0]['Accession_Number']
    val_patient_ids = data[data['Valid'] == 1]['Accession_Number']
    train_data = data[data['Accession_Number'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Accession_Number'].isin(val_patient_ids)].reset_index(drop=True)
    
    bags_train = create_bags(train_data, min_bag_size, max_bag_size, cropped_images, label_columns, instance_columns, instance_data)
    bags_val = create_bags(val_data, min_bag_size, max_bag_size, cropped_images, label_columns, instance_columns, instance_data)
    
    bags_train = upsample_minority_class(bags_train)  # Upsample the minority class in the training set
    
    print(f'There are {len(bags_train)} files in the training data')
    print(f'There are {len(bags_val)} files in the validation data')
    count_bag_labels(bags_train)
    
    
    
    save_bags_to_csv(bags_train, 'F:/Temp_SSD_Data/bags_testing.csv')
    
    return bags_train, bags_val
import os
import requests
import tarfile
import csv
from tqdm import tqdm
import random
import shutil
import pandas as pd

def download_and_extract_imagenette(output_dir, export_name):
    """Download and extract Imagenette dataset if not already present"""
    if not os.path.exists(f'{output_dir}/imagenette2.tgz'):
        print("Downloading Imagenette dataset...")
        url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
        response = requests.get(url, stream=True)
        with open(f'{output_dir}/imagenette2.tgz', 'wb') as file:
            shutil.copyfileobj(response.raw, file)

    if not os.path.exists(f'{output_dir}/{export_name}'):
        print("Extracting Imagenette dataset...")
        with tarfile.open(f'{output_dir}/imagenette2.tgz', 'r:gz') as tar_ref:
            tar_ref.extractall(f'{output_dir}/{export_name}')
    
    return f'{output_dir}/{export_name}/imagenette2/train'

def process_images(source_folder, output_dir, export_name):
    """Process images from source folder and create image_labels.csv"""
    
    # Set up the directories
    images_dir = f'{output_dir}/{config["export_name"]}/images'
    os.makedirs(images_dir, exist_ok=True)
    
    # Create the raw data csv file
    with open(f'{output_dir}/{export_name}/image_labels.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Label'])
        
        # File extensions to look for
        extensions = ('.jpg', '.jpeg', '.png', '.JPEG')
        
        # Get total number of files for the progress bar
        total_files = sum([len([f for f in files if f.lower().endswith(extensions)]) 
                          for r, d, files in os.walk(source_folder)])
        
        # Move all images to the images directory and write image labels to the CSV file
        with tqdm(total=total_files, desc="Processing images") as pbar:
            for label_dir in os.listdir(source_folder):
                label_path = os.path.join(source_folder, label_dir)
                if os.path.isdir(label_path):
                    for file in os.listdir(label_path):
                        if file.lower().endswith(extensions):
                            src_path = os.path.join(label_path, file)
                            
                            # create unique filename and copy files
                            new_filename = f"{label_dir}_{file}"
                            dst_path = os.path.join(images_dir, new_filename)
                            shutil.copy2(src_path, dst_path)
                            filename = new_filename
                            
                            writer.writerow([filename, label_dir])
                            pbar.update(1)
        
        
def create_mil_dataset(output_dir, export_name, target_label, label_name, 
                      positive_percentage, min_bag_size, max_bag_size):
    """Create Multiple Instance Learning dataset from processed images"""
    
    # Read the CSV file
    df = pd.read_csv(f'{output_dir}/{export_name}/image_labels.csv')
    image_labels = dict(zip(df['Image'], df['Label']))

    # Create the train.csv file and store bag information
    train_data = []
    bag_info = {}
    bag_id = 0
    all_images = list(image_labels.keys())
    random.shuffle(all_images)
    
    with tqdm(total=len(all_images), desc="Creating train.csv") as pbar:
        while all_images:
            bag_size = random.randint(min_bag_size, max_bag_size)
            bag_images = all_images[:bag_size]
            all_images = all_images[bag_size:]
            
            # Check if bag contains target label
            has_target = target_label in [image_labels[image_name] for image_name in bag_images]
            is_valid = random.random() < 0.2  # 20% probability of being in the validation set
            train_data.append([int(is_valid), bag_images, bag_id, has_target, bag_id])
            
            for image_name in bag_images:
                bag_info[image_name] = bag_id
            
            bag_id += 1
            pbar.update(bag_size)

    # Create the InstanceData.csv file
    instance_data = []
    selected_bags = random.sample(range(bag_id), int(bag_id * positive_percentage))
    
    for image_name, label in image_labels.items():
        bag_id = bag_info[image_name]
        if bag_id in selected_bags:
            has_target = label == target_label
            instance_data.append([bag_id, int(has_target), image_name])

    # Create DataFrames
    train_df = pd.DataFrame(train_data, columns=['Valid', 'Images', 'ID', label_name, 'Accession_Number'])
    instance_df = pd.DataFrame(instance_data, columns=['Accession_Number', label_name, 'ImageName'])

    # Write DataFrames to CSV files
    train_df.to_csv(f'{output_dir}/{export_name}/TrainData.csv', index=False)
    instance_df.to_csv(f'{output_dir}/{export_name}/InstanceData.csv', index=False)

    print(f"Dataset {config['export_name']} formatting completed.")

# Configuration for different datasets
DATASETS = {
    'imagenette_hard': {
        'export_name': 'imagenette_hard_test',
        'source_type': 'download',  # 'download' or 'local'
        'source_folder': None,  # Not needed for download
        'target_label': 'n01440764',
        'label_name': 'Has_Fish',
        'positive_percentage': 0.2, # percent of known positive instances
        'min_bag_size': 2,
        'max_bag_size': 10,
    },
    'imagenette_dog_hard': {
        'export_name': 'imagenette_dog_hard_test',
        'source_type': 'local',
        'source_folder': 'D:/DATA/CASBUSI/exports/ImageNet_Dog_Raw', # manually picked classes (all dogs)
        'target_label': 'n02098286',
        'label_name': 'Has_Highland',
        'positive_percentage': 0.2, # percent of known positive instances
        'min_bag_size': 2,
        'max_bag_size': 10,
    },
}



# Choose which dataset to process
output_dir = "D:/DATA/CASBUSI/exports"
config = DATASETS['imagenette_dog_hard']  # Change this line to select dataset

print(f"Processing dataset: {config['export_name']}")

# Determine source folder based on dataset type
if config['source_type'] == 'download':
    source_folder = download_and_extract_imagenette(output_dir, config['export_name'])
else:
    source_folder = config['source_folder']

# Process images
process_images(source_folder, output_dir, config['export_name'])

# Create MIL dataset
create_mil_dataset(
    output_dir, 
    config['export_name'],
    config['target_label'],
    config['label_name'],
    config['positive_percentage'],
    config['min_bag_size'],
    config['max_bag_size']
)
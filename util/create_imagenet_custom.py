import os
import requests
import tarfile
import csv
from tqdm import tqdm
import random
import shutil
import pandas as pd

output_dir = "D:\DATA\CASBUSI\exports"
export_name = "imagenette_dog_hard"
source_folder = "D:/DATA/CASBUSI/exports/ImageNet_Dog_Raw"  # Add your source folder path here
positive_percentage = .2 #0.2
min_bag_size = 2
max_bag_size = 10
target_label = 'n02098286'
label_name = 'Has_Highland'

# Set up the directories
images_dir = f'{output_dir}/{export_name}/images'
os.makedirs(images_dir, exist_ok=True)

# Create the raw data csv file
with open(f'{output_dir}/{export_name}/image_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Label'])
    
    # Get total number of files for the progress bar
    total_files = sum([len(files) for r, d, files in os.walk(source_folder) if any(f.endswith(('.jpg', '.jpeg', '.png', '.JPEG')) for f in files)])
    
    # Move all images to the images directory and write image labels to the CSV file
    with tqdm(total=total_files, desc="Processing images") as pbar:
        for label_dir in os.listdir(source_folder):
            label_path = os.path.join(source_folder, label_dir)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.jpeg')):
                        src_path = os.path.join(label_path, file)
                        # Create a unique filename to avoid conflicts
                        new_filename = f"{label_dir}_{file}"
                        dst_path = os.path.join(images_dir, new_filename)
                        shutil.copy2(src_path, dst_path)  # Using copy2 instead of move
                        writer.writerow([new_filename, label_dir])
                        pbar.update(1)


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
        has_fish = target_label in [image_labels[image_name] for image_name in bag_images]
        is_valid = random.random() < 0.2  # 20% probability of being in the validation set
        train_data.append([int(is_valid), bag_images, bag_id, has_fish, bag_id])
        
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
        has_fish = label == target_label
        instance_data.append([bag_id, int(has_fish), image_name])

# Create DataFrames
train_df = pd.DataFrame(train_data, columns=['Valid', 'Images', 'ID', label_name, 'Accession_Number'])
instance_df = pd.DataFrame(instance_data, columns=['Accession_Number', label_name, 'ImageName'])

# Write DataFrames to CSV files
train_df.to_csv(f'{output_dir}/{export_name}/TrainData.csv', index=False)
instance_df.to_csv(f'{output_dir}/{export_name}/InstanceData.csv', index=False)

print("Dataset formatting completed.")
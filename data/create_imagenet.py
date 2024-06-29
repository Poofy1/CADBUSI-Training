import os
import requests
import tarfile
import csv
import random
import shutil

output_dir = "D:\DATA\CASBUSI\exports"
export_name = "imagenette2_insane"
positive_percentage = 0.1  # Percentage of positive instances to include 


if not os.path.exists(f'{output_dir}/imagenette2.tgz'):
    # Download the Imagenette dataset
    url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
    response = requests.get(url, stream=True)
    with open(f'{output_dir}/imagenette2.tgz', 'wb') as file:
        shutil.copyfileobj(response.raw, file)

if not os.path.exists(f'{output_dir}/{export_name}'):
    # Extract the dataset
    with tarfile.open(f'{output_dir}/imagenette2.tgz', 'r:gz') as tar_ref:
        tar_ref.extractall(f'{output_dir}/{export_name}')

# Delete the .tgz file
#os.remove(f'{output_dir}/imagenette2.tgz')

# Set up the directories
images_dir = f'{output_dir}/{export_name}/images'
train_dir = f'{output_dir}/{export_name}/train'
os.makedirs(images_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)

# Create the intermediate CSV file to store image labels
with open(f'{output_dir}/{export_name}/image_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Label'])
    
    # Move all images to the images directory and write image labels to the CSV file
    for label_dir in os.listdir(f'{output_dir}/{export_name}/imagenette2/train'):
        label_path = os.path.join(f'{output_dir}/{export_name}/imagenette2/train', label_dir)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.endswith('.JPEG'):
                    src_path = os.path.join(label_path, file)
                    dst_path = os.path.join(images_dir, file)
                    shutil.move(src_path, dst_path)
                    writer.writerow([file, label_dir])

# Read the image labels from the intermediate CSV file
image_labels = {}
with open(f'{output_dir}/{export_name}/image_labels.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for row in reader:
        image_labels[row[0]] = row[1]

# Create the train.csv file and store bag information
train_data = []
bag_info = {}
bag_id = 0
all_images = list(image_labels.keys())
random.shuffle(all_images)
while all_images:
    bag_size = random.randint(2, 10)
    bag_images = all_images[:bag_size]
    all_images = all_images[bag_size:]
    has_fish = 'n01440764' in [image_labels[image_name] for image_name in bag_images]
    is_valid = random.random() < 0.2  # 20% probability of being in the validation set
    train_data.append([int(is_valid), bag_images, bag_id, has_fish, bag_id])
    
    for image_name in bag_images:
        bag_info[image_name] = bag_id
    
    bag_id += 1

# Create the InstanceData.csv file
instance_data = []
selected_bags = random.sample(range(bag_id), int(bag_id * positive_percentage))
for image_name, label in image_labels.items():
    bag_id = bag_info[image_name]
    if bag_id in selected_bags:
        has_fish = label == 'n01440764'
        accession_number = bag_id
        instance_data.append([accession_number, int(has_fish), image_name])

# Write the train.csv file
with open(f'{output_dir}/{export_name}/TrainData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Valid', 'Images', 'ID', 'Has_Fish', 'Accession_Number'])
    writer.writerows(train_data)

# Write the InstanceData.csv file
with open(f'{output_dir}/{export_name}/InstanceData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Accession_Number', 'Has_Fish', 'ImageName'])
    writer.writerows(instance_data)

print("Dataset formatting completed.")
from fastai.vision.all import *
import os
import shutil
import pandas as pd

def Move_Images(source_folder1, source_folder2, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Function to move .JPEG files from subfolders and delete the source folder
    def move_files(source_folder):
        # Check if the source folder exists
        if not os.path.exists(source_folder):
            return

        for subfolder in os.listdir(source_folder):
            subfolder_path = os.path.join(source_folder, subfolder)
            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    if filename.lower().endswith('.jpeg'):  # Check for .JPEG files
                        source_path = os.path.join(subfolder_path, filename)
                        destination_path = os.path.join(destination_folder, filename)
                        shutil.move(source_path, destination_path)

        # Delete the source folder after moving the files
        shutil.rmtree(source_folder)

    # Move .JPEG images from both source folders and delete them
    move_files(source_folder1)
    move_files(source_folder2)

    print(f".JPEG images moved to {destination_folder}")



def Export_Database(export_location, min_per_bag, max_per_bag, target_label):
    print("Reformatting Data:")

    # Directories
    csv_file = f'{export_location}noisy_imagenette.csv' 

    # Read data
    original_data = pd.read_csv(csv_file)

    # Split data into train and validation
    train_data = original_data[original_data['is_valid'] == False]
    val_data = original_data[original_data['is_valid'] == True]

    # Function to create bags
    def create_bags(data, is_valid):
        bags = []
        while not data.empty:
            bag_size = random.randint(min_per_bag, max_per_bag)
            bag_size = min(bag_size, len(data))  # Adjust bag size if fewer images are left
            sampled_data = data.sample(n=bag_size)
            data = data.drop(sampled_data.index)  # Remove selected images from the dataset

            has_label = target_label in sampled_data['noisy_labels_0'].values
            image_names = [os.path.basename(path) for path in sampled_data['path']]
            bags.append({'ID': len(bags) + 1, 'Images': str(image_names), 'Has_Label': has_label, 'Valid': is_valid})
        return bags

    # Create bags for train and validation
    train_bags = create_bags(train_data, 0)
    val_bags = create_bags(val_data, 1)

    # Combine bags and create DataFrame
    all_bags = train_bags + val_bags
    new_data = pd.DataFrame(all_bags)

    # Export to new CSV file
    new_csv_file = f'{export_location}TrainData.csv'
    new_data.to_csv(new_csv_file, index=False)

    print(f"Data exported to {new_csv_file}")





'''
# Define
n01440764='fish',
n02102040='dog',
n02979186='stereo',
n03000684='saw',
n03028079='church',
n03394916='horn',
n03417042='truck',
n03425413='pump',
n03445777='ball',
n03888257='chute'
'''


img_size = 160
download_location = f'D:/DATA/CASBUSI/exports/'
export_location = f'{download_location}/imagenette2-{img_size}/'


# Download data
path = untar_data(URLs.IMAGENETTE_160, data=download_location)
path_train = path/'train'
path_val = path/'val'

Move_Images(f'{export_location}/train', f'{export_location}/val', f'{export_location}/images')

Export_Database(export_location, 3, 7, 'n01440764')
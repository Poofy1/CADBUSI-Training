import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


def filter_raw_data(breast_data, image_data):

    # Join dataframes on PatientID
    data = pd.merge(breast_data, image_data, left_on=['Patient_ID', 'Breast'], right_on=['Patient_ID', 'laterality'], suffixes=('', '_image_data'))

    # Remove columns from image_data that also exist in breast_data
    for col in breast_data.columns:
        if col + '_image_data' in data.columns:
            data.drop(col + '_image_data', axis=1, inplace=True)
    
    
    data = data[data['Has_Unknown'] == False]

    # Reset the index
    data.reset_index(drop=True, inplace=True)
    
    return data


def create_bags(data, bag_size, root_dir):
    unique_patient_ids = data['Patient_ID'].unique()

    bags = []
    for patient_id in tqdm(unique_patient_ids):
        bag_files = []
        bag_labels = []

        patient_data = data[data['Patient_ID'] == patient_id]
        
        # Exclude bags that exceed the bag_size
        if len(patient_data) > bag_size:
            continue

        # If less than bag_size, randomly choose rows for upsampling until we reach bag_size
        while len(bag_files) < bag_size:
            row = patient_data.sample(n=1).iloc[0]
            filename = os.path.join(root_dir, row['ImageName'])
            label = int(row['Has_Malignant'])
            bag_files.append(filename)
            bag_labels.append(label)

        bag_id = [patient_id] * bag_size

        bags.append([bag_files, bag_labels, bag_id])

    # Identify minority and majority class bags
    malignant_bags = [bag for bag in bags if sum(bag[1]) > 0]
    benign_bags = [bag for bag in bags if sum(bag[1]) == 0]

    if len(malignant_bags) < len(benign_bags):
        minority_bags = malignant_bags
        label_value = 1
    else:
        minority_bags = benign_bags
        label_value = 0

    # Create new bags for the minority class
    diff = len(bags) - 2 * len(minority_bags) 
    new_patient_id = max(unique_patient_ids) + 1

    for _ in range(diff):
        chosen_bag = random.choice(minority_bags)
        new_bag_files = random.sample(chosen_bag[0], bag_size)
        new_bag_labels = [label_value] * bag_size
        new_bag_id = [new_patient_id] * bag_size

        bags.append([new_bag_files, new_bag_labels, new_bag_id])
        new_patient_id += 1

    return bags

def count_malignant_bags(bags):
    malignant_count = 0
    non_malignant_count = 0
    
    for bag in bags:
        bag_labels = bag[1]  # Extracting labels from the bag
        if sum(bag_labels) > 0:  # If there's even one malignant instance
            malignant_count += 1
        else:
            non_malignant_count += 1
    
    return malignant_count, non_malignant_count

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
    
    

def upsample_bag_to_min_count(group, min_count):
    num_needed = min_count - len(group)
    
    if num_needed > 0:
        # Duplicate random samples within the same group
        random_rows = group.sample(num_needed, replace=True)
        group = pd.concat([group, random_rows], axis=0).reset_index(drop=True)
        
    return group
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.utils import resample
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle


def filter_raw_data(breast_data, image_data):

    # Join dataframes on PatientID
    data = pd.merge(breast_data, image_data, left_on=['Accession_Number', 'Breast'], right_on=['Accession_Number', 'laterality'], suffixes=('', '_image_data'))

    # Remove columns from image_data that also exist in breast_data
    for col in breast_data.columns:
        if col + '_image_data' in data.columns:
            data.drop(col + '_image_data', axis=1, inplace=True)
    
    
    data = data[data['Has_Unknown'] == False]

    # Reset the index
    data.reset_index(drop=True, inplace=True)
    
    data = upsample_minority(data)
    
    return data

def upsample_minority(data):
    # Group data by ['Accession_Number', 'Breast'] and get the first entry for 'Has_Malignant' to determine the class
    grouped_data = data.groupby(['Accession_Number', 'Breast']).agg({'Has_Malignant': 'first'})
    
    # Determine the minority class (1 for malignant, 0 for benign)
    minority_class = 1 if grouped_data['Has_Malignant'].mean() < 0.5 else 0
    
    # Filter the groups belonging to the minority and majority classes
    minority_groups = grouped_data[grouped_data['Has_Malignant'] == minority_class].index.tolist()
    majority_groups = grouped_data[grouped_data['Has_Malignant'] != minority_class].index.tolist()
    
    # Determine the difference in count between majority and minority classes
    count_diff = len(majority_groups) - len(minority_groups)
    
    # If the classes are already balanced, return the original data
    if count_diff == 0:
        return data
    
    # Select groups from the minority class to duplicate
    minority_data = data[data.set_index(['Accession_Number', 'Breast']).index.isin(minority_groups)]
    
    # Create an iterator to cycle through the minority_data groups
    group_cycle = cycle(minority_data.groupby(['Accession_Number', 'Breast']))
    
    # Duplicate the necessary number of groups from the minority class to achieve balance
    duplicated_data = pd.concat([next(group_cycle)[1] for _ in range(count_diff)], ignore_index=True)
    
    # Find the maximum Accession_Number to generate new Accession_Numbers for duplicated rows
    max_acc_number = data['Accession_Number'].max()
    
    # Generate new Accession_Numbers for the duplicated rows
    new_acc_numbers = range(max_acc_number + 1, max_acc_number + 1 + count_diff)
    acc_number_mapping = dict(zip(duplicated_data['Accession_Number'].unique(), new_acc_numbers))
    duplicated_data['Accession_Number'] = duplicated_data['Accession_Number'].map(acc_number_mapping)
    
    # Concatenate the original data with the duplicated data
    upsampled_data = pd.concat([data, duplicated_data], ignore_index=True)
    
    return upsampled_data



def create_bags(data, min_size, max_size, root_dir):
    unique_patient_ids = data['Accession_Number'].unique()

    bag_files = []
    bag_labels = []
    bag_ids = []
    id = 0  # initialize bag id

    for patient_id in tqdm(unique_patient_ids):
        patient_data = data[data['Accession_Number'] == patient_id]

        # Exclude bags that are outside the size range
        if not (min_size <= len(patient_data) <= max_size):
            continue

        bag_file = []  # temporary lists to hold file names and labels for this bag
        bag_label = []

        for _, row in patient_data.iterrows():
            filename = os.path.join(root_dir, row['ImageName'])
            label = int(row['Has_Malignant'])
            bag_file.append(filename)
            bag_label.append(label)

        bag_files.append(np.array(bag_file))  # convert to numpy array and append to bag_files
        bag_labels.append(np.array(bag_label))  # convert to numpy array and append to bag_labels
        bag_ids.append(id * np.ones(len(bag_file), dtype=int))  # create id array and append to bag_ids

        id += 1  # increment bag id for the next bag

    return bag_files, bag_labels, bag_ids



def count_bag_labels(bag_labels):
    positive_bag_count = 0
    negative_bag_count = 0

    for labels in bag_labels:
        # Assuming a bag is positive if it contains at least one positive instance
        if np.any(labels == 1):
            positive_bag_count += 1
        else:
            negative_bag_count += 1

    return positive_bag_count, negative_bag_count

def process_single_image(row, root_dir, output_dir, resize_and_pad):
    try:
        img_name = row['ImageName']
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
    
    
    
def prepare_all_data(export_location, case_study_data, breast_data, image_data, cropped_images, img_size, min_bag_size, max_bag_size):
    
    print("Preprocessing Data...")
    data = filter_raw_data(breast_data, image_data)
    
    #Cropping images
    preprocess_and_save_images(data, export_location, cropped_images, img_size)
    
    # Split the data into training and validation sets
    train_patient_ids = case_study_data[case_study_data['valid'] == 0]['Patient_ID']
    val_patient_ids = case_study_data[case_study_data['valid'] == 1]['Patient_ID']
    train_data = data[data['Patient_ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Patient_ID'].isin(val_patient_ids)].reset_index(drop=True)
    
    #data.to_csv(f'{env}/testData.csv')
    
    bags_train, bags_train_labels_all, bags_train_ids = create_bags(train_data, min_bag_size, max_bag_size, cropped_images)
    bags_val, bags_val_labels_all, bags_val_ids = create_bags(val_data, min_bag_size, max_bag_size, cropped_images)
    
    files_train = np.concatenate( bags_train )
    ids_train = np.concatenate( bags_train_ids )
    labels_train = np.array([1 if np.any(x == 1) else 0 for x in bags_train_labels_all], dtype=np.float32)

    files_val = np.concatenate( bags_val )
    ids_val = np.concatenate( bags_val_ids )
    labels_val = np.array([1 if np.any(x == 1) else 0 for x in bags_val_labels_all], dtype=np.float32)
    
    print(f'There are {len(files_train)} files in the training data')
    print(f'There are {len(files_val)} files in the validation data')
    malignant_count, non_malignant_count = count_bag_labels(labels_train)
    print(f"Number of Malignant Bags: {malignant_count}")
    print(f"Number of Non-Malignant Bags: {non_malignant_count}")
    
    val_patient_ids_list = val_data['Patient_ID'].tolist()
    train_patient_ids_list = val_data['Patient_ID'].tolist()
    return files_train, ids_train, labels_train, train_patient_ids_list, files_val, ids_val, labels_val, val_patient_ids_list
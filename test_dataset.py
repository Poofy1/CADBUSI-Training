import os
from fastai.vision.all import *
from torch.nn.functional import binary_cross_entropy
from model_ABMIL import *
from train_loop import *
from data_prep import *
import matplotlib.pyplot as plt
env = os.path.dirname(os.path.abspath(__file__))



def load_trained_model(model_path, encoder_arch):
    encoder = create_timm_body(encoder_arch)
    nf = num_features_model(nn.Sequential(*encoder.children()))
    aggregator = ABMIL_aggregate(nf=nf, num_classes=1, pool_patches=3, L=128)
    bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
    bagmodel.load_state_dict(torch.load(model_path))
    bagmodel.eval()
    return bagmodel




def predict_on_test_set(model, test_dl):
    bag_predictions = []
    
    with torch.no_grad():
        for (data, yb) in tqdm(test_dl, total=len(test_dl)): 
            xb, yb = data, yb.cuda()

            outputs = model(xb).squeeze(dim=1)
            
            # For the bag-level prediction, we can take the maximum prediction across images in the bag.
            bag_pred = torch.max(outputs).item()
            bag_predictions.append(bag_pred)
    
    return bag_predictions


def compute_errors(predictions, true_labels):
    return [binary_cross_entropy(torch.tensor([pred]), torch.tensor([label])).item() for pred, label in zip(predictions, true_labels)]


def generate_test_filenames_from_folders(test_root_path):
    bag_files = []
    bag_ids = []
    current_id = 0
    
    for subdir, _, files in os.walk(test_root_path):
        if files:
            current_bag = [os.path.join(subdir, f) for f in files if f.endswith(('.jpg', '.png'))]
            bag_files.extend(current_bag)
            
            # Assign the same ID for all images in this bag
            bag_ids.extend([current_id] * len(current_bag))
            
            current_id += 1  # Increment the ID for the next bag
    
    return bag_files, bag_ids


def test_dataset():
    # Load data
    files_train, ids_train, labels_train, train_patient_ids, files_val, ids_val, labels_val, val_patient_ids = prepare_all_data(export_location, case_study_data, breast_data, image_data, 
                                                                                                                            cropped_images, img_size, min_bag_size, max_bag_size)

    # Combine training and validation data
    combined_files = np.concatenate((files_train, files_val))
    combined_ids = np.concatenate((ids_train, ids_val))
    combined_labels = np.concatenate((labels_train, labels_val))
    combined_patient_ids = np.concatenate((train_patient_ids, val_patient_ids))

    # Now use the combined data for the dataset
    dataset_combined = BagOfImagesDataset(combined_files, combined_ids, combined_labels, train=False)
    combined_dl = TUD.DataLoader(dataset_combined, batch_size=1, collate_fn=collate_custom, drop_last=True)


    # Make predictions on test set
    predictions = predict_on_test_set(model, combined_dl)

    # Compute errors
    errors = compute_errors(predictions, combined_labels)

    # Sort errors to get worst cases
    sorted_indices = np.argsort(errors)[::-1]  # Sorting in descending order

    # Create an empty DataFrame
    df_failed_cases = pd.DataFrame(columns=["Patient_ID", "Prediction", "True_Label", "Error"])

    # Create a list to store the records
    failed_cases_records = []

    # Populate the list with the worst-performing cases data
    for idx in sorted_indices:
        patient_id = combined_patient_ids[idx]
        pred = round(predictions[idx], 4)
        true_label = combined_labels[idx]
        error = round(errors[idx], 4)
        
        # Append this record to the list
        failed_cases_records.append({"Patient_ID": patient_id, "Prediction": pred, "True_Label": true_label, "Error": error})

    # Convert the list of records to a DataFrame
    df_failed_cases = pd.DataFrame(failed_cases_records)

    # Save the DataFrame to a CSV file
    mkdir(f"{env}/tests/", exist_ok=True)
    output_path = f"{env}/tests/failed_cases.csv"
    df_failed_cases.to_csv(output_path, index=False)
    print(f"Saved failed cases")
    
    return df_failed_cases


# Config
model_name = 'NoMixup3'
encoder_arch = 'resnet18'
img_size = 350
min_bag_size = 3
max_bag_size = 15

# Paths
export_location = 'D:/DATA/CASBUSI/exports/export_10_28_2023/'
case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
image_data = pd.read_csv(f'{export_location}/ImageData.csv')
cropped_images = f"F:/Temp_SSD_Data/{img_size}_images/"


# Load the trained model
model_path = f'{env}/models/{model_name}/{model_name}.pth'
model = load_trained_model(model_path, encoder_arch)

if False:
    df_failed_cases = test_dataset()
else:
    df_failed_cases = pd.read_csv(f"{env}/tests/failed_cases.csv")


# Merge df_failed_cases with case_study_data to get the BI-RADS scores
merged_data = pd.merge(df_failed_cases, case_study_data[['Patient_ID', 'BI-RADS']], on='Patient_ID', how='left')

# Group by BI-RADS scores and calculate average error for each group
average_errors = merged_data.groupby('BI-RADS')['Error'].mean()

# Sort by BI-RADS score (which is the index after grouping)
average_errors = average_errors.sort_index()

# Plot the average errors for each BI-RADS score
plt.figure(figsize=(12, 8))
average_errors.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Error by BI-RADS Score')
plt.xlabel('BI-RADS Score')
plt.ylabel('Average Error')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as an image
image_save_path = f"{env}/tests/BI-RADS_average_error.png"
plt.savefig(image_save_path)

print(f"Saved the plot to {image_save_path}")



# Count number of rows/images for each Patient_ID in image_data
image_counts = image_data.groupby('Patient_ID').size().reset_index(name='Image_Count')

# Merge with df_failed_cases to get errors for each Patient_ID
merged_data_images = pd.merge(df_failed_cases, image_counts, on='Patient_ID', how='left')

# Group by Image_Count and calculate average error for each group
average_errors_images = merged_data_images.groupby('Image_Count')['Error'].mean()

# Sort by Image_Count (which is the index after grouping)
average_errors_images = average_errors_images.sort_index()

# Plot the average errors against number of images
plt.figure(figsize=(12, 8))
average_errors_images.plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Average Error by Number of Images per Case')
plt.xlabel('Number of Images')
plt.ylabel('Average Error')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as an image
image_save_path_images = f"{env}/tests/Average_Error_by_Image_Count.png"
plt.savefig(image_save_path_images)

print(f"Saved the plot to {image_save_path_images}")

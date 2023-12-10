import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from archs.model_ABMIL import *
from train_ABMIL import *
from data.format_data import *
import matplotlib.pyplot as plt


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


def predict_on_test_set(bagmodel, test_dl):
    loss_func = nn.BCELoss()
    
    bag_predictions = []
    bag_losses = []
    bag_ids = [] 
    bag_labels = []
    
    with torch.no_grad():
        for (data, yb, bag_id) in tqdm(test_dl, total=len(test_dl)): 
            xb, yb = data, yb.cuda()
            
            outputs, _, _, _  = bagmodel(xb)
            outputs = outputs[0][0]
            loss = loss_func(outputs, yb[0][0])

            bag_predictions.append(round(outputs.cpu().item(), 4))
            bag_losses.append(round(loss.cpu().item(), 4))
            bag_ids.append(bag_id.cpu().item())
            
            # Convert yb to a list of labels
            bag_labels.append(yb[0].cpu().numpy().tolist())

    return bag_predictions, bag_losses, bag_ids, bag_labels


def test_dataset(output_path, label_columns):
    # Load data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, cropped_images, img_size, min_bag_size, max_bag_size)

    # Combine training and validation data
    combined_dict = bags_train
    combined_dict.update(bags_val)

    # Now use the combined data for the dataset
    #dataset_combined = TUD.Subset(BagOfImagesDataset(combined_dict, train=False),list(range(0,100)))
    dataset_combined = BagOfImagesDataset(combined_dict, train=False)
    combined_dl = TUD.DataLoader(dataset_combined, batch_size=1, collate_fn=collate_custom, drop_last=True)

    # Make predictions on test set
    predictions, losses, bag_ids, bag_labels = predict_on_test_set(bagmodel, combined_dl)
    
    # Convert bag_labels to a DataFrame with separate columns
    labels_df = pd.DataFrame(bag_labels, columns=label_columns)

    # Combine the labels DataFrame with the other data
    results_df = pd.concat([
        pd.DataFrame({"Accession_Number": bag_ids, "Prediction": predictions, "Loss": losses}),
        labels_df
    ], axis=1)
    
    # Sort the DataFrame by Loss column in descending order
    results_df = results_df.sort_values(by="Loss", ascending=False)

    # Save the DataFrame to a CSV file
    results_df.to_csv(f'{output_path}/failed_cases.csv', index=False)
    print("Saved failed cases")
    
    return results_df


def plot_and_save_average_errors(average_errors, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(12, 8))
    average_errors.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved the plot to {save_path}")
    
    
# Config
model_name = 'NoMixup_11_14_2'
dataset_name = 'export_11_11_2023'
label_columns = ['Has_Malignant', 'Has_Benign']
encoder_arch = 'resnet18'
img_size = 350
min_bag_size = 2
max_bag_size = 20

# Paths
export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
output_path = f"{parent_dir}/results/{model_name}_DatasetTest/"
case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
image_data = pd.read_csv(f'{export_location}/ImageData.csv')

mkdir(output_path, exist_ok=True)

# Load the trained model
model_path = f'{parent_dir}/models/{model_name}/{model_name}.pth'
encoder = create_timm_body(encoder_arch)
nf = num_features_model(nn.Sequential(*encoder.children()))
aggregator = ABMIL_aggregate(nf=nf, num_classes=1, pool_patches=3, L=128)
bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
bagmodel.load_state_dict(torch.load(model_path))
bagmodel.eval()



if True:
    df_failed_cases = test_dataset(output_path, label_columns)
else:
    df_failed_cases = pd.read_csv(f"{output_path}/failed_cases.csv")


# Merge df_failed_cases with case_study_data to get the BI-RADS scores
merged_data = pd.merge(df_failed_cases, case_study_data[['Accession_Number', 'BI-RADS']], on='Accession_Number', how='left')

# Group by BI-RADS scores and calculate average error for each group
average_errors = merged_data.groupby('BI-RADS')['Loss'].mean()

# Sort by BI-RADS score (which is the index after grouping)
average_errors = average_errors.sort_index()

plot_and_save_average_errors(average_errors, 'Average Loss by BI-RADS Score', 'BI-RADS Score', 'Average Loss', f"{output_path}/BI-RADS_average_loss.png")

# Save the plot as an image
image_save_path = f"{parent_dir}/results/BI-RADS_average_loss.png"
plt.savefig(image_save_path)



# Count number of rows/images for each Accession_Number in image_data
image_counts = image_data.groupby('Accession_Number').size().reset_index(name='Image_Count')

# Merge with df_failed_cases to get errors for each Accession_Number
merged_data_images = pd.merge(df_failed_cases, image_counts, on='Accession_Number', how='left')

# Group by Image_Count and calculate average error for each group
average_errors_images = merged_data_images.groupby('Image_Count')['Loss'].mean()

# Sort by Image_Count (which is the index after grouping)
average_errors_images = average_errors_images.sort_index()

plot_and_save_average_errors(average_errors_images, 'Average Loss by Number of Images per Accession_Number', 'Number of Images', 'Average Loss', f"{output_path}/Average_Loss_by_Image_Count.png")

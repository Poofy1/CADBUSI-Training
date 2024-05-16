import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from archs.model_ABMIL import *
#from train_ABMIL import *
from train_GenSCL_ITS2CLR import *
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
        for (data, yb, instance_yb, id) in tqdm(test_dl, total=len(test_dl)): 
            xb, yb = data, yb.cuda()
            
            outputs, _, _  = bagmodel(xb, pred_on = True)
            #outputs = outputs[0][0]
            loss = loss_func(outputs, yb)

            bag_predictions.append(round(outputs.cpu().item(), 4))
            bag_losses.append(round(loss.cpu().item(), 4))
            bag_ids.append(id.cpu().item())
            
            # Convert yb to a list of labels
            bag_labels.append(yb[0].cpu().numpy().tolist())

    return bag_predictions, bag_losses, bag_ids, bag_labels


def test_dataset(output_path, label_columns, instance_columns):
    # Load data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    
    train_data = pd.read_csv(f'{export_location}/TrainData.csv')

    # Combine training and validation data
    combined_dict = bags_train
    combined_dict.update(bags_val)

    
    val_transform = T.Compose([
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    # Now use the combined data for the dataset
    #dataset_combined = TUD.Subset(BagOfImagesDataset(combined_dict, train=False),list(range(0,100)))
    dataset_combined = BagOfImagesDataset(combined_dict, transform=val_transform)
    combined_dl = TUD.DataLoader(dataset_combined, batch_size=1, collate_fn=collate_bag, drop_last=True)

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

    #Replace Accession_Number in results_df using map
    id_to_acc_number = pd.Series(train_data.Accession_Number.values,index=train_data.ID).to_dict()
    results_df['Accession_Number'] = results_df['Accession_Number'].map(id_to_acc_number)
    
    # Save the DataFrame to a CSV file
    results_df.to_csv(f'{output_path}/bag_predictions.csv', index=False)
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
    
    

if __name__ == '__main__':

    # Config
    model_name = '03_18_2024_Res50_Head_5'
    encoder_arch = 'resnet50'
    dataset_name = 'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = []#['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present'] # 'Reject Image' is used to remove images and is not trained on
    img_size = 300
    batch_size = 1
    min_bag_size = 2
    max_bag_size = 25
    lr = 0.001

    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    case_study_data = pd.read_csv(f"{export_location}/CaseStudyData.csv")
    image_data = pd.read_csv(f"{export_location}/ImageData.csv")
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    output_path = f"{parent_dir}/results/{model_name}_DatasetTest/"
    mkdir(output_path, exist_ok=True)

    # Get Training Data
    num_labels = len(label_columns)


    encoder = create_timm_body(encoder_arch)
    nf = num_features_model( nn.Sequential(*encoder.children()))
    
    """aggregator = ABMIL_aggregate( nf = nf, num_classes = num_labels, pool_patches = 6, L = 128)
    bagmodel = EmbeddingBagModel(encoder, aggregator, num_classes = num_labels).cuda()"""
    
    bagmodel = Embeddingmodel(encoder = encoder, nf = nf, num_classes = num_labels, efficient_net = False).cuda()

    
    # Check if the model already exists
    model_folder = f"{parent_dir}/models/{model_name}/"
    model_path = f'{model_folder}/{model_name}.pth'
    bagmodel.load_state_dict(torch.load(model_path))
    print(f"Loaded pre-existing model from {model_name}")



    # Check if the bag_predictions.csv file exists
    bag_predictions_path = f"{output_path}/bag_predictions.csv"
    if os.path.exists(bag_predictions_path):
        df_failed_cases = pd.read_csv(bag_predictions_path)
    else:
        df_failed_cases = test_dataset(output_path, label_columns, instance_columns)

    # Merge df_failed_cases with case_study_data to get the BI-RADS scores
    merged_data = pd.merge(df_failed_cases, case_study_data[['Accession_Number', 'BI-RADS']], on='Accession_Number', how='left')

    # Group by BI-RADS scores and calculate average loss for each group
    average_errors = merged_data.groupby('BI-RADS')['Loss'].mean()
    average_errors = average_errors.sort_index()
    plot_and_save_average_errors(average_errors, 'Average Loss by BI-RADS Score', 'BI-RADS Score', 'Average Loss', f"{output_path}/BI-RADS_average_loss.png")

    # Count number of rows/images for each Accession_Number in image_data
    image_counts = image_data.groupby('Accession_Number').size().reset_index(name='Image_Count')

    # Merge with df_failed_cases to get errors for each Accession_Number
    merged_data_images = pd.merge(df_failed_cases, image_counts, on='Accession_Number', how='left')

    # Group by Image_Count and calculate average loss for each group
    average_errors_images = merged_data_images.groupby('Image_Count')['Loss'].mean()
    average_errors_images = average_errors_images.sort_index()
    plot_and_save_average_errors(average_errors_images, 'Average Loss by Number of Images per Accession_Number', 'Number of Images', 'Average Loss', f"{output_path}/Average_Loss_by_Image_Count.png")

    # Investigate high loss cases
    high_loss_threshold = 10.0  # Adjust this threshold as needed
    high_loss_cases = df_failed_cases[df_failed_cases['Loss'] > high_loss_threshold]
    print(f"Number of high loss cases (loss > {high_loss_threshold}): {len(high_loss_cases)}")
    print("High loss cases:")
    print(high_loss_cases)

    # Investigate cases with large number of images
    large_image_count_threshold = 20  # Adjust this threshold as needed
    large_image_count_cases = merged_data_images[merged_data_images['Image_Count'] > large_image_count_threshold]
    print(f"Number of cases with large number of images (count > {large_image_count_threshold}): {len(large_image_count_cases)}")
    print("Cases with large number of images:")
    print(large_image_count_cases)

    # Save the evaluation results
    evaluation_results_path = f"{output_path}/evaluation_results.txt"
    with open(evaluation_results_path, 'w') as f:
        f.write(f"Number of high loss cases (loss > {high_loss_threshold}): {len(high_loss_cases)}\n")
        f.write("High loss cases:\n")
        f.write(high_loss_cases.to_string(index=False))
        f.write("\n\n")
        f.write(f"Number of cases with large number of images (count > {large_image_count_threshold}): {len(large_image_count_cases)}\n")
        f.write("Cases with large number of images:\n")
        f.write(large_image_count_cases.to_string(index=False))
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from archs.model_ABMIL import *
from train_GenSCL_ITS2CLR import *
from data.format_data import *
import matplotlib.pyplot as plt

val_transform = T.Compose([
            CLAHETransform(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def test_dataset(output_path, label_columns, instance_columns):
    # Load data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)

    # Combine training and validation data
    #combined_dict = bags_train
    #combined_dict.update(bags_val)
    combined_dict = bags_train

    # Now use the combined data for the dataset
    #dataset_combined = TUD.Subset(BagOfImagesDataset(combined_dict, transform=val_transform, save_processed=False),list(range(0,50)))
    dataset_combined = BagOfImagesDataset(combined_dict, transform=val_transform, save_processed=False)
    combined_dl = TUD.DataLoader(dataset_combined, batch_size=1, collate_fn = collate_bag, drop_last=True)
    
    bag_data = {}
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (data, yb, instance_yb, id) in tqdm(combined_dl, total=len(combined_dl)): 
            xb, yb = data, yb.cuda()
            
            outputs, instance_pred, _ = model(xb, pred_on = True)
            print(instance_pred)
            
            loss = criterion(outputs, yb)
            bag_id = id.item()
            
            if bag_id not in bag_data:
                bag_data[bag_id] = {
                    'instance_predictions': [],
                    'instance_yb': [],
                    'output': [],
                    'losses': [],
                    'labels': [],
                }

                bag_data[bag_id]['losses'].append(round(loss.item(), 5))
                bag_data[bag_id]['output'].append(round(outputs.item(), 5))
                bag_data[bag_id]['labels'].append(yb[0].item())
                bag_data[bag_id]['instance_predictions'].append([round(x.item(), 4) for x in instance_pred[0]])
                bag_data[bag_id]['instance_yb'].append([int(x[0].item()) for x in instance_yb[0]])
    
    # Save the bag_data dictionary to disk
    with open(f'{output_path}/bag_data.pkl', 'wb') as f:
        pickle.dump(bag_data, f)
    
    return bag_data
    

if __name__ == '__main__':

    # Config
    model_name = 'export_03_18_2024_resnet50_03'
    dataset_name = 'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']   #['Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 300
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  30
    use_efficient_net = False
    
    model_name = "imagenette2_resnet18_02"
    dataset_name = 'imagenette2'
    label_columns = ['Has_Fish']
    instance_columns = []  
    img_size = 128
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    use_efficient_net = False
    
    model_folder = f"{env}/models/{model_name}/"
    lr = 0.001


    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    output_path = f"{parent_dir}/results/{model_name}_DatasetTest/"
    mkdir(output_path, exist_ok=True)

    # Get Training Data
    num_labels = len(label_columns)

    


    if os.path.exists(f'{output_path}/bag_data.pkl'):
        with open(f'{output_path}/bag_data.pkl', 'rb') as f:
            bag_data = pickle.load(f)
    else:
        
        # Create Model
        model = build_model(config)    
        
        
        
        
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")        
            
        BCE_loss = nn.BCELoss()
        genscl = GenSupConLossv2(temperature=0.07, contrast_mode='all', base_temperature=0.07)
        
        # Check if the model already exists
        model_folder = f"{parent_dir}/models/{model_name}/"
        model_path = f'{model_folder}/{model_name}.pth'
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pre-existing model from {model_name}")


        bag_data = test_dataset(output_path, label_columns, instance_columns)
        
    
    
    
    ###################################
    
    
    # Save the bag_data as a CSV file
    csv_file = f'{output_path}/bag_data.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header row
        writer.writerow(['Bag ID', 'Instance Predictions', 'Instance Labels', 'Output', 'Loss', 'Label'])
        
        # Write the data rows
        for bag_id, bag in bag_data.items():
            instance_predictions = [str(pred) for pred in bag['instance_predictions']]
            instance_labels = [str(label) for label in bag['instance_yb']]
            outputs = bag['output']
            losses = bag['losses']
            labels = [str(label) for label in bag['labels']]
            
            # Assuming all lists have the same length, iterate over the instances
            for i in range(len(instance_predictions)):
                row = [
                    bag_id,
                    instance_predictions[i],
                    instance_labels[i],
                    outputs[i],
                    losses[i],
                    labels[i]
                ]
                writer.writerow(row)

    print(f"CSV file saved at: {csv_file}")




    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    train_data = pd.read_csv(f'{export_location}/TrainData.csv')

    # Remove rows from train_data where 'Accession_Number' does not exist in case_study_data
    train_data = train_data[train_data['Accession_Number'].isin(case_study_data['Accession_Number'])]

    # Merge train_data and case_study_data based on 'Accession_Number'
    merged_data = pd.merge(train_data, case_study_data, on='Accession_Number')

    # Create a dictionary to store the errors for each BI-RADS type
    birads_errors = {}

    # Iterate over all the bags in the bag_data dictionary
    for bag_id, bag_info in bag_data.items():
        # Retrieve the corresponding BI-RADS type from the merged dataframe
        matching_rows = merged_data[merged_data['ID'] == bag_id]
        if not matching_rows.empty and 'BI-RADS' in matching_rows.columns:
            birads = matching_rows['BI-RADS'].values[0]
            # Add the error to the respective BI-RADS type in the dictionary
            if birads not in birads_errors:
                birads_errors[birads] = []
            birads_errors[birads].append(bag_info['losses'][0])

    # Calculate the average error for each BI-RADS type
    birads_avg_errors = {}
    for birads, errors in birads_errors.items():
        avg_error = sum(errors) / len(errors)
        birads_avg_errors[birads] = avg_error

    # Define the custom order for BI-RADS categories
    birads_order = ['0', '1', '2', '3', '4', '4A', '4B', '4C', '5', '6']

    # Create lists of BI-RADS labels and average errors in the custom order
    birads_labels = []
    avg_errors = []
    for birads in birads_order:
        if birads in birads_avg_errors:
            birads_labels.append(birads)
            avg_errors.append(birads_avg_errors[birads])

    # Create a bar graph of the average errors for each BI-RADS type
    plt.figure(figsize=(10, 6))
    plt.bar(birads_labels, avg_errors)
    plt.xlabel('BI-RADS')
    plt.ylabel('Average Error')
    plt.title('Average Error for Each BI-RADS Type')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the graph as an image file
    plt.savefig(f'{output_path}/birads_avg_errors.png')
    plt.close()
    
    
    
    ############

    # Create a list of tuples containing accession number, loss, and BI-RADS
    worst_bags = []
    for bag_id, bag_info in bag_data.items():
        matching_rows = merged_data[merged_data['ID'] == bag_id]
        if not matching_rows.empty and 'Accession_Number' in matching_rows.columns and 'BI-RADS' in matching_rows.columns:
            accession_number = matching_rows['Accession_Number'].values[0]
            birads = matching_rows['BI-RADS'].values[0]
            loss = bag_info['losses'][0]
            worst_bags.append((accession_number, loss, birads))

    # Sort the worst_bags list based on the loss in descending order
    worst_bags.sort(key=lambda x: x[1], reverse=True)

    print("Worst Performing Bags:")
    for accession_number, loss, birads in worst_bags[:5]:  # Print the top 5 worst performing bags
        print(f"Accession Number: {accession_number}, Loss: {loss:.4f}, BI-RADS: {birads}")
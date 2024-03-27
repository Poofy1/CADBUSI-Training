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
    combined_dict = bags_train
    combined_dict.update(bags_val)

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
            #print(outputs)
            
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

                bag_data[bag_id]['losses'].append(loss.item())
                bag_data[bag_id]['output'].append(outputs.item())
                bag_data[bag_id]['labels'].append(yb.cpu().numpy().tolist())
                bag_data[bag_id]['instance_predictions'].append(instance_pred)
                bag_data[bag_id]['instance_yb'].append(instance_yb)
    
    # Save the bag_data dictionary to disk
    with open(f'{output_path}/instance_data.pkl', 'wb') as f:
        pickle.dump(bag_data, f)
    
    return bag_data
    

if __name__ == '__main__':

    # Config
    model_name = '03_18_2024_Res18_01'
    dataset_name = 'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']   #['Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 300
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  30
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

    


    if os.path.exists(f'{output_path}/instance_data.pkl'):
        with open(f'{output_path}/instance_data.pkl', 'rb') as f:
            bag_data = pickle.load(f)
    else:
        
        # Get Model
        if use_efficient_net:
            encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            nf = 512
            # Replace the last fully connected layer with a new one
            num_features = encoder.classifier[1].in_features
            encoder.classifier[1] = nn.Linear(num_features, nf)
            
        else:
            encoder = create_timm_body("resnet18")
            nf = num_features_model( nn.Sequential(*encoder.children()))
        

        model = Embeddingmodel(encoder = encoder, nf = nf, num_classes = num_labels, efficient_net = use_efficient_net).cuda()
        
        
        
        
        
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
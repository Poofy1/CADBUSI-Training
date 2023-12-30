import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from archs.model_ABMIL import *
from train_ABMIL import *
from data.format_data import *
import matplotlib.pyplot as plt


def predict_on_test_set(bagmodel, test_dl, threshold=0.5):
    bag_predictions = []
    bag_labels = []
    
    with torch.no_grad():
        for (data, yb, bag_id) in tqdm(test_dl, total=len(test_dl)): 
            xb, yb = data, yb.cuda()
            
            outputs, _, _, _  = bagmodel(xb)
            outputs = outputs[0][0]
            
            binary_outputs = (outputs.cpu() > threshold).int()

            bag_predictions.append(binary_outputs.numpy().tolist())
            
            # Convert yb to a list of labels
            bag_labels.append(yb[0].cpu().numpy().tolist())
            
    # Convert predictions and labels to DataFrame
    predictions_df = pd.DataFrame(bag_predictions, columns=label_columns)
    labels_df = pd.DataFrame(bag_labels, columns=label_columns)

    return predictions_df, labels_df



# Calculate the 4x4 confusion matrix
def calculate_confusion_matrix(labels_df, predictions_df):
    combined = labels_df.astype(str) + predictions_df.astype(str)
    unique_combinations = sorted(set(combined['Has_Malignant'] + combined['Has_Benign']))
    confusion_matrix = pd.DataFrame(0, index=unique_combinations, columns=unique_combinations)
    
    for true, pred in zip(combined['Has_Malignant'] + combined['Has_Benign'], combined['Has_Malignant'] + combined['Has_Benign']):
        confusion_matrix.loc[true, pred] += 1
        
    return confusion_matrix
    
    
    
# Config
model_name = 'ABMIL_12_26_1'
encoder_arch = 'resnet18'
dataset_name = 'export_12_26_2023'
label_columns = ['Has_Malignant', 'Has_Benign']
img_size = 350
batch_size = 5
min_bag_size = 2
max_bag_size = 20
lr = 0.001

# Paths
export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
output_path = f"{parent_dir}/results/{model_name}_DatasetTest/"
mkdir(output_path, exist_ok=True)

# Load the trained model
model_path = f'{parent_dir}/models/{model_name}/{model_name}.pth'
encoder = create_timm_body(encoder_arch)
nf = num_features_model(nn.Sequential(*encoder.children()))
num_labels = len(label_columns)
aggregator = ABMIL_aggregate(nf=nf, num_classes=num_labels, pool_patches=6, L=128)
bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
bagmodel.load_state_dict(torch.load(model_path))
bagmodel.eval()



    
    
# Load data
bags_train, bags_val = prepare_all_data(export_location, label_columns, cropped_images, img_size, min_bag_size, max_bag_size)

# Now use the combined data for the dataset
dataset_val = TUD.Subset(BagOfImagesDataset(bags_val, train=False),list(range(0,100)))
#dataset_val = BagOfImagesDataset(bags_val, train=False)
combined_dl = TUD.DataLoader(dataset_val, batch_size=1, collate_fn=collate_custom, drop_last=True)

# Make predictions on test set
predictions_df, labels_df = predict_on_test_set(bagmodel, combined_dl)

conf_matrix = calculate_confusion_matrix(labels_df, predictions_df)
print(conf_matrix)
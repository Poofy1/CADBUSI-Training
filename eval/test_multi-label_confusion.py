import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from archs.model_ABMIL import *
from train_ABMIL import *
from data.format_data import *
import matplotlib.pyplot as plt
import seaborn as sns



# Calculate confusion matrix for a single label
def calculate_single_label_conf_matrix(labels, predictions):
    """
    Create a 2x2 confusion matrix for a single label.
    """
    conf_matrix = confusion_matrix(labels, predictions)
    return pd.DataFrame(conf_matrix, index=["Actual Negative", "Actual Positive"],
                        columns=["Predicted Negative", "Predicted Positive"])

    
def plot_confusion_matrix(conf_matrix, output_file):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_file)
    plt.close()
    
    
if __name__ == '__main__':

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
    output_path = f"{parent_dir}/results/{model_name}_Matrix/"
    mkdir(output_path, exist_ok=True)

    # Get Training Data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_labels = len(label_columns)

    # Create datasets
    dataset_val = TUD.Subset(BagOfImagesDataset(bags_val),list(range(0,100)))
    #dataset_val = BagOfImagesDataset(bags_val, train=False)

    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)

    encoder = create_timm_body(encoder_arch)
    nf = num_features_model( nn.Sequential(*encoder.children()))
    aggregator = ABMIL_aggregate( nf = nf, num_classes = num_labels, pool_patches = 6, L = 128)
    bagmodel = EmbeddingBagModel(encoder, aggregator, num_classes = num_labels).cuda()

    
    # Check if the model already exists
    model_folder = f"{parent_dir}/models/{model_name}/"
    model_path = f'{model_folder}/{model_name}.pth'
    bagmodel.load_state_dict(torch.load(model_path))
    print(f"Loaded pre-existing model from {model_name}")
    
    
    # EVALUATE MODEL
    bagmodel.eval()
    total = 0
    correct = [0] * num_labels
    bag_predictions = []
    bag_labels = []
    with torch.no_grad():
        for (data, yb, _) in tqdm(val_dl, total=len(val_dl)): 
            xb, yb = data, yb.cuda()
            outputs, _, _, _ = bagmodel(xb)
            predicted = (outputs > .5).float()

            # Append predictions and labels to respective lists
            bag_predictions.extend(predicted.cpu().numpy())
            bag_labels.extend(yb.cpu().numpy())
            
    
    # Creating separate confusion matrices for each label
    for label_idx, label in enumerate(label_columns):
        # Extracting individual label predictions and ground truths
        label_predictions = [pred[label_idx] for pred in bag_predictions]
        label_labels = [lbl[label_idx] for lbl in bag_labels]

        # Calculate confusion matrix for the current label
        label_conf_matrix = calculate_single_label_conf_matrix(label_labels, label_predictions)

        # Path to save the confusion matrix image
        conf_matrix_image_path = os.path.join(output_path, f'confusion_matrix_{label}.png')

        # Plot and save the confusion matrix
        plot_confusion_matrix(label_conf_matrix, conf_matrix_image_path)
        print(f"Confusion matrix for {label} saved as {conf_matrix_image_path}")
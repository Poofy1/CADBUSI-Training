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
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score


# Calculate confusion matrix for a single label
def calculate_single_label_conf_matrix(labels, predictions):
    conf_matrix = confusion_matrix(labels, predictions)
    # Swap columns when creating the DataFrame
    df = pd.DataFrame(conf_matrix, index=["Actual Negative", "Actual Positive"],
                      columns=["Predicted Negative", "Predicted Positive"])
    # Reorder columns to desired order
    return df[["Predicted Positive", "Predicted Negative"]]


def compute_metrics(labels, predictions):
    # Calculate the confusion matrix and extract components
    conf_matrix = confusion_matrix(labels, predictions)
    TN, FP, FN, TP = conf_matrix.ravel()
    
    # Calculate metrics
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    return sensitivity, specificity, precision, recall, f1
    
    
def plot_confusion_matrix(conf_matrix, label, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{label} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    plt.close()
    
    
if __name__ == '__main__':

    # Config
    model_name = 'ABMIL_12_26_1'
    encoder_arch = 'resnet18'
    dataset_name = 'export_12_26_2023'
    label_columns = ['Has_Malignant', 'Has_Benign']
    instance_columns = ['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
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
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_labels = len(label_columns)

    # Create datasets
    #dataset_val = TUD.Subset(BagOfImagesDataset(bags_val),list(range(0,100)))
    dataset_val = BagOfImagesDataset(bags_val, train=False)

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
        for (data, yb, instance_yb, _) in tqdm(val_dl, total=len(val_dl)): 
            xb, yb = data, yb.cuda()
            outputs, _, _, _ = bagmodel(xb)
            predicted = (outputs > .5).float()

            # Append predictions and labels to respective lists
            bag_predictions.extend(predicted.cpu().numpy())
            bag_labels.extend(yb.cpu().numpy())
            
    # Creating separate confusion matrices for each label and printing metrics
    for label_idx, label in enumerate(label_columns):
        # Extracting individual label predictions and ground truths
        label_predictions = [int(pred[label_idx] > 0.5) for pred in bag_predictions]
        label_labels = [lbl[label_idx] for lbl in bag_labels]

        # Compute metrics
        sensitivity, specificity, precision, recall, f1 = compute_metrics(label_labels, label_predictions)

        # Print out the metrics
        print(f"Metrics for {label}:")
        print(f"  Sensitivity (Recall): {sensitivity:.3f}")
        print(f"  Specificity: {specificity:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  F1 Score: {f1:.3f}")

        # Path to save the confusion matrix image
        conf_matrix_image_path = os.path.join(output_path, f'confusion_matrix_{label}.png')

        # Plot and save the confusion matrix
        plot_confusion_matrix(confusion_matrix(label_labels, label_predictions), label, conf_matrix_image_path)
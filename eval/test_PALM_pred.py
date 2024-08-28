import torch
import os
import sys
import torch.utils.data as TUD
from torchvision import transforms as T
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from util.format_data import *
from util.sudo_labels import *
from archs.save_arch import *
from archs.model_PALM2_solo import *
from data.bag_loader import *
from data.instance_loader import *

def run_model_and_save_predictions(model, dataloader, device, csvwriter, category):
    model.eval()
    
    with torch.no_grad():
        for images, labels, _, bag_ids in tqdm(dataloader, desc=f"Processing {category} bags"):
            bag_pred, _, _, _ = model(images, pred_on=True)
            bag_pred = bag_pred.squeeze()  # Ensure predictions are 1D
            labels = labels.squeeze().to(device)  # Ensure labels are 1D and on the correct device
            
            # Calculate loss per sample
            losses = F.binary_cross_entropy(bag_pred, labels, reduction='none')
            
            # Handle both batch and single-item cases
            if bag_pred.dim() == 0:  # Single item
                csvwriter.writerow([
                    int(bag_ids.item()),
                    round(bag_pred.item(), 4),
                    round(losses.item(), 4),
                    int(labels.item()),
                    category
                ])
            else:  # Batch
                for bag_id, pred, label, loss_val in zip(bag_ids, bag_pred, labels, losses):
                    csvwriter.writerow([
                        int(bag_id.item()),
                        round(pred.item(), 4),
                        round(loss_val.item(), 4),
                        int(label.item()),
                        category
                    ])
                
                    
def run_test(dataset_name, label_columns, instance_columns, config, output_file):
    img_size = config['img_size']
    bag_batch_size = config['bag_batch_size']
    min_bag_size = config['min_bag_size']
    max_bag_size = config['max_bag_size']
    arch = config['arch']
    pretrained_arch = config['pretrained_arch']
    num_labels = len(label_columns)

    # Define transforms
    transform = T.Compose([
        CLAHETransform(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare data
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    bags_train, bags_test = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)

    # Create datasets and dataloaders
    dataset_train = BagOfImagesDataset(bags_train, transform=transform, save_processed=False)
    dataset_test = BagOfImagesDataset(bags_test, transform=transform, save_processed=False)
    
    dataloader_train = TUD.DataLoader(dataset_train, batch_size=bag_batch_size, collate_fn=collate_bag, shuffle=False)
    dataloader_test = TUD.DataLoader(dataset_test, batch_size=bag_batch_size, collate_fn=collate_bag, shuffle=False)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(arch, pretrained_arch, num_classes=num_labels).to(device)
    
    # Load the saved model state
    if model_version:
        model_path = f"{model_folder}/{head_name}/{model_version}/model.pth"
    else:
        model_path = f"{model_folder}/{head_name}/model.pth"
    model.load_state_dict(torch.load(model_path))

    # Run model on train and test sets
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Accession_Number', 'Prediction', 'Loss', 'Has_Malignant', 'Category'])
        
        print(f"Processing train set...")
        run_model_and_save_predictions(model, dataloader_train, device, csvwriter, 'train')
        
        print(f"Processing test set...")
        run_model_and_save_predictions(model, dataloader_test, device, csvwriter, 'val')

if __name__ == '__main__':
    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  

    # Load the model configuration
    head_name = "Head_Palm4_CASBUSI_2_efficientnet_b0"
    model_version = "1" #Leave "" to read HEAD
    
    # loaded configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)
    
    dataset = config['dataset_name'] #'export_03_18_2024' 

    # Run test and save predictions
    output_file = os.path.join(current_dir, f"{dataset}_predictions.csv")
    run_test(dataset, config['label_columns'], config['instance_columns'], config, output_file)

    print(f"Predictions have been saved to: {output_file}")
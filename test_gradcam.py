import os
from fastai.vision.all import *
from torch.nn.functional import binary_cross_entropy
from model_FC import *
from train_loop_FC import *
from data_prep import *
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
env = os.path.dirname(os.path.abspath(__file__))


def load_trained_model(model_path, encoder_arch):
    encoder = create_timm_body(encoder_arch)
    nf = num_features_model(nn.Sequential(*encoder.children()))
    aggregator = FC_aggregate(nf=nf, num_classes=1, L=128, fc_layers=[256, 128], dropout = .5)
    bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
    bagmodel.load_state_dict(torch.load(model_path))
    bagmodel.eval()
    return bagmodel

            
def predict_on_test_set(model, test_dl, save_path):
    loss_func = nn.BCELoss()
    
    bag_predictions = []
    bag_losses = []
    bag_ids = [] 
    bag_labels = []
    
    target_layer = [model.encoder[7][1].conv2]
    
    grad_cam = GradCAMPlusPlus(model, target_layers=target_layer)
    
    torch.set_grad_enabled(True)
    
    for (data, yb, bag_id) in tqdm(test_dl, total=len(test_dl)): 
        xb, yb = data, yb.cuda()

        outputs, _, _ = model(xb)
        loss = loss_func(outputs, yb)
        
        bag_predictions.append(round(outputs.cpu().item(), 4))
        bag_losses.append(round(loss.cpu().item(), 4))
        bag_ids.append(bag_id[0].cpu().numpy())  # assuming bag_id is a tensor
        bag_labels.append(yb.cpu().item())

        
        # Process each image in the batch
        for i in range(xb.shape[0]):
            input_tensor = xb[i].unsqueeze(0)  # Add batch dimension

            # Use GradCAM++ to generate the CAM
            targets = [ClassifierOutputTarget(0)]
            grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            # Convert tensor image to numpy array
            rgb_img = xb[i].cpu().numpy().transpose(1, 2, 0)
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())  # Normalize

            # Overlay the CAM on the image
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # Save or display the visualization
            plt.imshow(visualization)
            plt.title(f'Bag ID: {bag_id[i]}, Prediction: {outputs[i].item():.4f}')
            plt.savefig(f'{save_path}/cam_{bag_id[i]}.png')
            plt.close()


        return bag_predictions, bag_losses, bag_ids, bag_labels


def test_dataset():
    test_location = f"{env}/tests/{model_name}/"
    os.makedirs(test_location, exist_ok=True)

    # Load data
    bags_train, bags_val = prepare_all_data(export_location, case_study_data, breast_data, image_data, cropped_images, img_size, min_bag_size, max_bag_size)

    # Now use the combined data for the dataset
    #dataset_combined = TUD.Subset(BagOfImagesDataset( combined_files, combined_ids, combined_labels),list(range(0,100)))
    dataset_val = BagOfImagesDataset(bags_val, train=False)
    combined_dl = TUD.DataLoader(dataset_val, batch_size=1, collate_fn=collate_custom, drop_last=True)

    # Make predictions on test set
    predictions, losses, bag_ids, bag_labels = predict_on_test_set(model, combined_dl, test_location)
    
    
    
    # Create a DataFrame to save the results
    results_df = pd.DataFrame({
        "Accession_Number": bag_ids,
        "Prediction": predictions,
        "True_Label": bag_labels,
        "Loss": losses
    })

    return results_df


# Config
model_name = 'FC_test'
encoder_arch = 'resnet18'
img_size = 350
min_bag_size = 2
max_bag_size = 20
export_location = 'D:/DATA/CASBUSI/exports/export_11_11_2023/'

# Paths
case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
image_data = pd.read_csv(f'{export_location}/ImageData.csv')
cropped_images = f"F:/Temp_SSD_Data/{img_size}_images/"

# Load the trained model
model_path = f'{env}/models/{model_name}/{model_name}.pth'
model = load_trained_model(model_path, encoder_arch)

# Test a validation cases
results = test_dataset()

# Merge df_failed_cases with case_study_data to get the BI-RADS scores
merged_data = pd.merge(results, case_study_data[['Accession_Number', 'BI-RADS']], on='Accession_Number', how='left')


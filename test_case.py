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


def predict_on_test_set(model, test_dl, save_path):
    loss_func = nn.BCELoss()
    
    bag_predictions = []
    bag_losses = []
    bag_ids = [] 
    bag_labels = []
    saliency_maps_list = []  # to store saliency maps
    
    with torch.no_grad():
        for (data, yb, bag_id) in tqdm(test_dl, total=len(test_dl)): 
            xb, yb = data, yb.cuda()
            
            outputs, saliency_maps, yhat, att = model(xb)
            loss = loss_func(outputs, yb)

            
def predict_on_test_set(model, test_dl, save_path):
    loss_func = nn.BCELoss()
    
    bag_predictions = []
    bag_losses = []
    bag_ids = [] 
    bag_labels = []
    saliency_maps_list = []  # to store saliency maps
    
    with torch.no_grad():
        for (data, yb, bag_id) in tqdm(test_dl, total=len(test_dl)): 
            xb, yb = data, yb.cuda()
            
            outputs, saliency_maps, yhat, att = model(xb)
            loss = loss_func(outputs, yb)

            # Saving the images with saliency maps overlaid
            for bag_index, bag_saliency_maps in enumerate(saliency_maps):
                for i, saliency in enumerate(bag_saliency_maps):
                    # Squeeze the saliency tensor to remove any singleton dimensions
                    saliency = saliency.squeeze()

                    # Normalize the saliency map for visualization
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

                    # Create an empty RGB image with the same size as the saliency map
                    saliency_map_rgb = torch.zeros(3, saliency.size(0), saliency.size(1))

                    # Place the saliency map in the red channel
                    saliency_map_rgb[0] = saliency

                    # Convert the RGB saliency map to a PIL image
                    saliency_img = TF.to_pil_image(saliency_map_rgb.cpu().detach())

                    # Retrieve the corresponding image tensor from the batch
                    img_tensor = xb[bag_index][i].cpu()

                    # Unnormalize the image tensor to convert it to PIL image
                    img = TF.to_pil_image(unnormalize(img_tensor).cpu().detach())

                    # Ensure the saliency_img is the same size as the original image
                    saliency_img = saliency_img.resize(img.size, Image.BILINEAR)

                    # Overlay the red saliency map on the image
                    overlayed_img = Image.blend(img, saliency_img, alpha=0.5)

                    # Save the overlayed image
                    overlayed_img.save(os.path.join(save_path, f'saliency_bag_{bag_id[bag_index]}_img_{i}.png'))


            bag_predictions.append(round(outputs.cpu().item(), 4))
            bag_losses.append(round(loss.cpu().item(), 4))
            bag_ids.append(bag_id[0].cpu().numpy())  # assuming bag_id is a tensor
            bag_labels.append(yb.cpu().item())
            
            # Convert each tensor in saliency_maps to numpy and store in saliency_maps_list
            saliency_maps_list.extend([s.cpu().numpy() for s in saliency_maps])  # Adjusted this line

        return bag_predictions, bag_losses, bag_ids, bag_labels, saliency_maps_list


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
    predictions, losses, bag_ids, bag_labels, saliency_maps = predict_on_test_set(model, combined_dl, test_location)
    
    
    
    # Create a DataFrame to save the results
    results_df = pd.DataFrame({
        "Accession_Number": bag_ids,
        "Prediction": predictions,
        "True_Label": bag_labels,
        "Loss": losses
    })

    return results_df


# Config
model_name = 'Mixup_11_13_23'
encoder_arch = 'resnet50'
img_size = 325
min_bag_size = 3
max_bag_size = 13
export_location = 'D:/DATA/CASBUSI/exports/export_10_28_2023/'

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


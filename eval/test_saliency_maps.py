import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from archs.model_ABMIL import *
from train_ABMIL import *
from data.format_data import *
from PIL import ImageDraw, ImageFont

def draw_legend(image, label_columns, true_labels, pred_labels):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # You can choose a different font or size

    legend_height = 20  # Adjust as needed
    text_start = 10 + len(label_columns) * legend_height

    # Draw color boxes and label text
    for i, label in enumerate(label_columns):
        color = (255, 0, 0) if i == 0 else (0, 255, 0)
        draw.rectangle([(10, i * legend_height + 10), (30, i * legend_height + 30)], fill=color)
        draw.text((35, i * legend_height + 10), label, fill=(255, 255, 255), font=font)

    # Format and draw true labels
    true_labels_str = '\n'.join([f"{label_columns[i]}: {true_labels[i].item()}" for i in range(len(label_columns))])
    draw.multiline_text((10, text_start + 10), f"True Labels:\n{true_labels_str}", fill=(255, 255, 255), font=font)

    # Format and draw predicted labels (rounded to 2 decimal places)
    pred_labels_str = '\n'.join([f"{label_columns[i]}: {pred_labels[i].item():.2f}" for i in range(len(label_columns))])
    draw.multiline_text((10, text_start + 20 + 20 * len(label_columns)), f"Predicted Labels:\n{pred_labels_str}", fill=(255, 255, 255), font=font)
    
    
    
if __name__ == '__main__':

    # Config
    model_name = 'ABMIL_12_26_1'
    encoder_arch = 'resnet18'
    dataset_name = 'export_12_26_2023'
    label_columns = ['Has_Malignant', 'Has_Benign']
    img_size = 350
    batch_size = 1
    min_bag_size = 2
    max_bag_size = 20
    lr = 0.001

    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    output_path = f"{parent_dir}/results/{model_name}_Map/"
    mkdir(output_path, exist_ok=True)

    # Get Training Data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, cropped_images, img_size, min_bag_size, max_bag_size)
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
    
    
    bagmodel.eval()
    
    
    bag_predictions = []
    bag_losses = []
    bag_ids = [] 
    bag_labels = []
    saliency_maps_list = []

    with torch.no_grad():
        for (data, yb, bag_id) in tqdm(val_dl, total=len(val_dl)): 
            xb, yb = data, yb.cuda()

            outputs, saliency_maps, yhat, att = bagmodel(xb)

            # Saving the images with saliency maps overlaid
            for bag_index, bag_saliency_maps in enumerate(saliency_maps):
                for i, saliency in enumerate(bag_saliency_maps):
                    # Initialize an empty RGB image
                    saliency_map_rgb = torch.zeros(3, saliency.size(1), saliency.size(2))

                    if saliency.shape[0] == num_labels and num_labels > 1:
                        # Normalize and assign each label's saliency map to a different color channel
                        for label_index in range(num_labels):
                            # Normalize the saliency map for the current label
                            saliency_label = (saliency[label_index] - saliency[label_index].min()) / (saliency[label_index].max() - saliency[label_index].min())
                            # Assign to red channel for first label, green for second
                            saliency_map_rgb[label_index] = saliency_label
                    else:
                        # Single label case, use red channel
                        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
                        saliency_map_rgb[0] = saliency

                    # Convert the RGB saliency map to a PIL image
                    saliency_img = TF.to_pil_image(saliency_map_rgb.cpu().detach())

                    # Retrieve the corresponding image tensor from the batch
                    img_tensor = xb[bag_index][i].cpu()

                    # Unnormalize the image tensor to convert it to PIL image
                    img = TF.to_pil_image(unnormalize(img_tensor).cpu().detach())

                    # Ensure the saliency_img is the same size as the original image
                    saliency_img = saliency_img.resize(img.size, Image.Resampling.BILINEAR)

                    # Overlay the saliency map on the image
                    overlayed_img = Image.blend(img, saliency_img, alpha=0.5)
                    
                    # Extract true and predicted labels for the current instance
                    current_true_labels = yb[bag_index]
                    current_predicted_labels = outputs[bag_index]

                    # Overlay the saliency map on the image and draw the legend
                    draw_legend(overlayed_img, label_columns, current_true_labels, current_predicted_labels)

                    # Save the overlayed image
                    overlayed_img.save(os.path.join(output_path, f'saliency_bag_{bag_id[bag_index]}_img_{i}.png'))

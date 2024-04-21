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
    dataset_name = 'export_03_18_2024'
    label_columns = ['Has_Malignant', 'Has_Benign']
    instance_columns = [] #['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present'] # 'Reject Image' is used to remove images and is not trained on
    img_size = 350
    batch_size = 1
    min_bag_size = 2
    max_bag_size = 25
    lr = 0.001

    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    output_path = f"{parent_dir}/results/{model_name}_Map/"
    mkdir(output_path, exist_ok=True)

    # Get Training Data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_labels = len(label_columns)
    
    val_transform = T.Compose([
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # Create datasets
    #dataset_val = TUD.Subset(BagOfImagesDataset(bags_val),list(range(0,100)))
    dataset_val = BagOfImagesDataset(bags_val, transform=val_transform)

    val_dl = TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)

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
        for (data, yb, instance_yb, bag_id) in tqdm(val_dl, total=len(val_dl)):
            xb, yb = data, yb.cuda()
            outputs, saliency_maps, yhat, att = bagmodel(xb)
            outputs = outputs[0][0]
            yb = yb[0][0]
            print(f'{outputs.item()} / {yb.item()}')

            for bag_index, bag_saliency_maps in enumerate(saliency_maps):
                bag_folder = os.path.join(output_path, f'bag_{bag_id[bag_index]}')
                os.makedirs(bag_folder, exist_ok=True)

                # Create a text file for the current bag
                bag_txt_file = os.path.join(bag_folder, 'bag_info.txt')
                with open(bag_txt_file, 'w') as f:
                    f.write(f'Label: {yb.item()}\n')
                    f.write(f'Bag Prediction: {outputs.item()}\n')

                for i, saliency in enumerate(bag_saliency_maps):
                    saliency_map_rgb = torch.zeros(3, saliency.size(1), saliency.size(2))
                    saliency = saliency[0]  # Use only the first label
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
                    saliency_map_rgb[0] = saliency  # Assign to the red channel

                    saliency_img = TF.to_pil_image(saliency_map_rgb.cpu().detach())
                    saliency_img = saliency_img.convert("RGBA")  # Convert to RGBA mode

                    # Make non-red pixels transparent
                    pixel_data = saliency_img.load()
                    for y in range(saliency_img.size[1]):
                        for x in range(saliency_img.size[0]):
                            r, g, b, a = pixel_data[x, y]
                            if r == 0 and g == 0 and b == 0:
                                pixel_data[x, y] = (0, 0, 0, 0)  # Set alpha channel to 0 for non-red pixels
                            else:
                                pixel_data[x, y] = (r, g, b, 128)  # Set alpha channel to 128 for red pixels

                    img_tensor = xb[bag_index][i].cpu()
                    img = TF.to_pil_image(unnormalize(img_tensor).cpu().detach())
                    saliency_img = saliency_img.resize(img.size, Image.Resampling.BILINEAR)

                    overlayed_img = Image.alpha_composite(img.convert("RGBA"), saliency_img)

                    original_img_path = os.path.join(bag_folder, f'original_{i}.png')
                    overlayed_img_path = os.path.join(bag_folder, f'overlayed_{i}.png')

                    img.save(original_img_path)
                    overlayed_img.convert("RGB").save(overlayed_img_path)
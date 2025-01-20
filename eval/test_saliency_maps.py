import os, sys
import torchvision.transforms.functional as TF

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from data.format_data import *
from data.sudo_labels import *
from loss.palm import PALM
from data.save_arch import *
from archs.model_solo_MIL_saliency import *
from data.bag_loader import *
from data.instance_loader import *
from PIL import ImageDraw, ImageFont
from util.eval_util import *


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

if __name__ == '__main__':

    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  
    
    # Load the model configuration
    head_name = "TEST1000"
    model_version = "" #Leave "" to read HEAD
    
    # loaded configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)

    # Paths
    output_path = get_metrics_path(head_name, model_version)

    # Get Training Data
    bags_train, bags_val, _, _ = prepare_all_data(config)
    num_labels = len(config['label_columns'])
    
    val_transform = T.Compose([
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # Create datasets
    dataset_val = BagOfImagesDataset(bags_val, transform=val_transform)
    val_dl = TUD.DataLoader(dataset_val, batch_size=1, collate_fn = collate_bag, drop_last=True)

    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes=num_labels).to(device)
    
    # Load the saved model state
    if model_version:
        model_path = f"{model_folder}/{head_name}/{model_version}/model.pth"
    else:
        model_path = f"{model_folder}/{head_name}/model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    
    bag_predictions = []
    bag_losses = []
    bag_ids = [] 
    bag_labels = []
    saliency_maps_list = []

    output_path = os.path.join(output_path, "saliency_maps")
    os.makedirs(output_path, exist_ok=True)

    # Create a CSV file to store all predictions
    csv_file_path = os.path.join(output_path, 'all_predictions.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow(['Bag ID', 'Image Index', 'True Label', 'Bag Prediction', 'Instance Prediction'])

        with torch.no_grad():
            for (data, yb, instance_yb, bag_id) in tqdm(val_dl, total=len(val_dl)):
                xb, yb = data, yb.cuda()
                bag_pred, bag_instance_predictions, instance_predictions, saliency_maps = model(xb, pred_on=True)
                
                #print(f"Saliency maps shape: {saliency_maps.shape}")
                
                for bag_index, bag in enumerate(xb):
                    for i in range(bag.size(0)):
                        # Get the saliency map for this image
                        saliency = saliency_maps[bag_index * bag.size(0) + i]

                        # Process saliency map
                        saliency = saliency.squeeze()  # Remove any extra dimensions
                        if len(saliency.shape) != 2:
                            print(f"Unexpected saliency shape: {saliency.shape}")
                            continue

                        saliency_map_rgb = torch.zeros(3, saliency.size(0), saliency.size(1))
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

                        # Save only the overlayed image
                        overlayed_img_path = os.path.join(output_path, f'overlayed_bag{bag_id[bag_index]}_img{i}.png')
                        overlayed_img.convert("RGB").save(overlayed_img_path)

                        # Write to CSV
                        instance_pred = instance_predictions[bag_index * bag.size(0) + i].item()
                        csv_writer.writerow([bag_id[bag_index], i, yb[bag_index].item(), bag_pred[bag_index].item(), instance_pred])

    print(f"CSV file with all predictions has been saved to {csv_file_path}")
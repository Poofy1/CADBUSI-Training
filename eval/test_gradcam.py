import os, sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from PIL import Image
import csv
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from data.format_data import *
from data.sudo_labels import *
from loss.palm import PALM
from data.save_arch import *
from archs.model_MIL import *
from data.bag_loader import *
from data.instance_loader import *

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        def save_activation(module, input, output):
            self.activations = output.detach().clone()
            
        def save_gradient(module, grad_input, grad_output):
            # Save the first derivative and ensure only positive gradients contribute.
            self.gradients = grad_output[0].detach().clone()
            self.gradients = F.relu(self.gradients)
            
        self.target_layer.register_forward_hook(save_activation)
        self.target_layer.register_full_backward_hook(save_gradient)
    
    def __call__(self, bags):
        all_bag_cams = []
        
        # Iterate over each bag
        for bag_idx, bag_images in enumerate(bags):
            bag_cams = []
            # Process each instance in the bag
            for instance_idx, image in enumerate(bag_images):
                self.model.zero_grad()
                
                # Add batch dimension if needed
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                
                # Get original image size
                _, _, height, width = image.shape
                
                # Forward pass (assumes model returns bag_pred, etc.)
                bag_pred, _, instance_predictions, _ = self.model([image], pred_on=True)

                # Backward pass
                bag_pred.backward(retain_graph=True)
                
                # Generate CAM using Grad-CAM++ formulation:
                gradients = self.gradients  # shape: [1, C, H', W']
                activations = self.activations  # shape: [1, C, H', W']
                
                # Compute squared and cubed gradients
                gradients_2 = gradients ** 2
                gradients_3 = gradients ** 3
                
                # Compute the denominator: 2 * grad^2 + sum_{i,j} (A * grad^3)
                denom = 2 * gradients_2 + torch.sum(activations * gradients_3, dim=(2,3), keepdim=True)
                # Avoid division by zero
                denom = torch.where(denom != 0, denom, torch.ones_like(denom))
                
                # Compute alpha coefficients per pixel
                alpha = gradients_2 / denom
                
                # Compute weights: sum over spatial locations of (alpha * ReLU(gradients))
                weights = torch.sum(alpha * F.relu(gradients), dim=(2,3), keepdim=True)
                
                # Generate the CAM by weighted combination of the activations
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
                
                # Post-process CAM: thresholding, upsampling, normalization
                cam = F.relu(cam - cam.mean() * 0.5)
                cam = F.interpolate(cam, size=(height, width), mode='bilinear', align_corners=False)
                
                cam_min = cam.min()
                cam_max = cam.max()
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)
                cam[cam < 0.3] = 0
                
                # Scale by confidence
                scaling_factor = torch.sigmoid(bag_pred)
                cam = cam * scaling_factor.view(1, 1, 1, 1)
                
                bag_cams.append(cam)
                
            all_bag_cams.append(torch.cat(bag_cams, dim=0))
        
        return all_bag_cams

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    out = tensor.clone()  # clone to ensure we don't modify the original
    for i, (m, s) in enumerate(zip(mean, std)):
        out[i] = out[i].mul(s).add(m)
    return out

def apply_heatmap(img_tensor, cam):
    # Convert image tensor to PIL Image
    img = TF.to_pil_image(unnormalize(img_tensor.squeeze().cpu()))
    
    # Ensure cam is the right shape and type
    cam = cam.squeeze()
    cam = (cam * 255).astype(np.uint8)
    
    # Make sure cam has a single channel
    if cam.ndim == 2:
        cam = cam[:, :, None]
    if cam.shape[-1] > 1:
        cam = cam[:, :, 0]
    
    # Create heatmap using OpenCV
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = Image.fromarray(heatmap).resize(img.size, Image.BILINEAR)
    
    # Adjust opacity based on activation strength
    opacity = np.maximum(cam.squeeze() / 255.0, 0.3)
    result = Image.blend(img.convert('RGB'), heatmap, float(opacity.mean()))
    
    return result

def disable_inplace_activations(module):
    """
    Recursively traverse the module and replace in-place activations with out-of-place ones.
    Checks for nn.ReLU and nn.SiLU activations.
    """
    for name, child in module.named_children():
        replaced = False
        if isinstance(child, nn.ReLU) and child.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
            replaced = True
        elif isinstance(child, nn.SiLU) and getattr(child, 'inplace', False):
            setattr(module, name, nn.SiLU(inplace=False))
            replaced = True

        if not replaced:
            disable_inplace_activations(child)

# Get the last convolutional layer from the encoder
def get_last_conv_layer(model):
    # Iterate through modules to find the last convolutional layer
    last_conv_layer = None
    for name, module in model.encoder.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
    return last_conv_layer


if __name__ == '__main__':
    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  
    head_name = "TEST300"
    model_version = "1"  # Leave "" to read HEAD
    
    # Load the model configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)

    # Get Training Data
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_labels = len(config['label_columns'])
    
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes=num_labels).to(device)
    
    # Load the saved model state
    if model_version:
        model_path = f"{model_folder}/{head_name}/{model_version}/model.pth"
        save_dir = f"{model_folder}/{config['head_name']}/{model_version}"
    else:
        model_path = f"{model_folder}/{head_name}/model.pth"
        save_dir = f"{model_folder}/{config['head_name']}"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    if "efficientnet" in config['arch'].lower():
        disable_inplace_activations(model.encoder)

    # Setup Grad-CAM++ using a target layer (adjust as needed)
    #target_layer = model.encoder[-1][-1]  # Example: last layer of encoder; adjust per architecture
    # Setup Grad-CAM
    target_layer = get_last_conv_layer(model)
    if target_layer is None:
        raise ValueError("Could not find a convolutional layer in the encoder")

    grad_cam = GradCAM(model, target_layer)

    # Create output directory
    output_path = os.path.join(save_dir, 'tests', 'GradCam')
    os.makedirs(output_path, exist_ok=True)

    # Create CSV file for predictions
    csv_file_path = os.path.join(output_path, 'all_predictions.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Bag ID', 'Image Index', 'True Label', 'Bag Prediction', 'Instance Prediction'])

        for (images, yb, instance_yb, bag_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)):
            # Clone tensors before moving to GPU
            if not isinstance(images, list):
                images = images.clone().cuda()
                yb = yb.clone().cuda()
            else:
                images = [img.clone().cuda() for img in images]
                yb = yb.clone().cuda()
                
            bag_pred, _, instance_predictions, _ = model(images, pred_on=True)
            cams = grad_cam(images)  # Get list of CAMs for all images

            for bag_index, bag in enumerate(images):
                for i in range(bag.size(0)):
                    img_tensor = bag[i].unsqueeze(0).to(device)
                    # Get corresponding CAM for this image
                    cam = cams[bag_index][i].squeeze().detach().cpu().numpy()
                    # Apply heatmap and save
                    result = apply_heatmap(img_tensor, cam)
                    result_path = os.path.join(output_path, f'gradcam_bag{bag_id[bag_index]}_img{i}.png')
                    result.save(result_path)
                    # Write to CSV
                    instance_pred = instance_predictions[bag_index][i]
                    csv_writer.writerow([bag_id[bag_index], i, yb[bag_index].item(), 
                                         bag_pred[bag_index].item(), instance_pred.item()])

    print(f"CSV file with all predictions has been saved to {csv_file_path}")

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from PIL import Image
import csv
from tqdm import tqdm

# Set up paths
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

class ScoreCAM:
    """
    Implementation of Score-CAM for visual explanations.
    Instead of using gradients, Score-CAM uses the confidence scores obtained by
    masking the input image with upsampled activation maps from a target convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.register_hook()
        
    def register_hook(self):
        def save_activation(module, input, output):
            # Save the activation map (detached so that no gradients are stored)
            self.activations = output.detach().clone()
        self.hook_handle = self.target_layer.register_forward_hook(save_activation)
    
    def __call__(self, bags):
        all_bag_cams = []
        
        # Process each bag (a list of instance images)
        for bag_idx, bag_images in enumerate(bags):
            bag_cams = []
            for instance_idx, image in enumerate(bag_images):
                # Ensure image has a batch dimension
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                _, _, height, width = image.shape
                
                # --- Forward pass on the original image ---
                # The forward hook will store the activations in self.activations.
                bag_pred, _, instance_predictions, _ = self.model([image], pred_on=True)
                # Save a copy of the activation maps from the original forward pass.
                original_activations = self.activations.clone()  # shape: [1, C, H', W']
                orig_bag_pred = bag_pred.clone()  # Used later for scaling the CAM
                
                # Temporarily remove the hook so that masked forward passes do not overwrite the stored activations.
                self.hook_handle.remove()
                
                num_channels = original_activations.shape[1]
                weights = []
                
                # --- Vectorized computation for all activation channels ---
                # original_activations: [1, num_channels, H', W']
                upsampled_masks = F.interpolate(original_activations, size=(height, width), mode='bilinear', align_corners=False)  
                # Now upsampled_masks has shape [1, num_channels, H, W]. Remove the batch dimension:
                upsampled_masks = upsampled_masks.squeeze(0)  # shape: [num_channels, H, W]

                # Normalize each activation map (mask) individually:
                m_min = upsampled_masks.view(upsampled_masks.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
                m_max = upsampled_masks.view(upsampled_masks.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
                normalized_masks = (upsampled_masks - m_min) / (m_max - m_min + 1e-7)  # shape: [num_channels, H, W]

                # Prepare a batch of masked images:
                # image shape: [1, C, H, W] -> expand to [num_channels, C, H, W]
                repeated_image = image.expand(normalized_masks.size(0), -1, -1, -1)
                # Expand normalized_masks to have the channel dimension: [num_channels, 1, H, W]
                normalized_masks = normalized_masks.unsqueeze(1)
                # Compute the masked images:
                masked_images = repeated_image * normalized_masks  # shape: [num_channels, C, H, W]

                # Forward pass all masked images at once:
                with torch.no_grad():
                    masked_preds, _, _, _ = self.model([masked_images], pred_on=True)
                    # Assume masked_preds is of shape [num_channels, 1] (or similar); compute weights:
                    weights = torch.sigmoid(masked_preds).view(-1)  # shape: [num_channels]

                # Now combine the original upsampled activations weighted by the computed weights:
                upsampled_activations = F.interpolate(original_activations, size=(height, width), mode='bilinear', align_corners=False)
                upsampled_activations = upsampled_activations.squeeze(0)  # shape: [num_channels, H, W]
                # Multiply each channel by its corresponding weight and sum:
                cam = (upsampled_activations * weights.view(-1, 1, 1)).sum(dim=0, keepdim=True).unsqueeze(0)  # shape: [1, 1, H, W]

                # --- Post-process CAM as before ---
                cam = F.relu(cam - cam.mean() * 0.5)
                cam = F.interpolate(cam, size=(height, width), mode='bilinear', align_corners=False)
                cam_min = cam.min()
                cam_max = cam.max()
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)
                cam[cam < 0.3] = 0

                # Optionally, scale by the original prediction confidence:
                scaling_factor = torch.sigmoid(orig_bag_pred)
                cam = cam * scaling_factor.view(1, 1, 1, 1)
                
                bag_cams.append(cam)
                # Re-register the hook for the next instance.
                self.register_hook()
            all_bag_cams.append(torch.cat(bag_cams, dim=0))
        
        return all_bag_cams

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    out = tensor.clone()  # Clone to avoid modifying the original tensor
    for i, (m, s) in enumerate(zip(mean, std)):
        out[i] = out[i].mul(s).add(m)
    return out

def apply_heatmap(img_tensor, cam):
    # Convert the image tensor to a PIL Image (after unnormalizing)
    img = TF.to_pil_image(unnormalize(img_tensor.squeeze().cpu()))
    
    # Ensure the CAM is a proper numpy array in the range [0, 255]
    cam = cam.squeeze()
    cam = (cam * 255).astype(np.uint8)
    
    # If cam has more than one channel, take the first channel
    if cam.ndim == 2:
        cam = cam[:, :, None]
    if cam.shape[-1] > 1:
        cam = cam[:, :, 0]
    
    # Create a heatmap using OpenCV’s COLORMAP_JET
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = Image.fromarray(heatmap).resize(img.size, Image.BILINEAR)
    
    # Adjust opacity based on activation strength (mean value of the CAM)
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

def get_last_conv_layer(model):
    """
    Traverse the model's encoder to find the last convolutional layer.
    """
    last_conv_layer = None
    for name, module in model.encoder.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_layer = module
    return last_conv_layer

if __name__ == '__main__':
    # Define paths and load model configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_folder = os.path.join(parent_dir, "models")  
    head_name = "TEST192"
    model_version = "1"  # Use a specific version or leave as "" to read HEAD
    
    # Load the model configuration.
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)

    # Get training data.
    bags_train, bags_val, bag_dataloader_train, bag_dataloader_val = prepare_all_data(config)
    num_labels = len(config['label_columns'])
    
    # Load the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes=num_labels).to(device)
    
    # Load the saved model state.
    if model_version:
        model_path = f"{model_folder}/{head_name}/{model_version}/model.pth"
    else:
        model_path = f"{model_folder}/{head_name}/model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    if "efficientnet" in config['arch'].lower():
        disable_inplace_activations(model.encoder)

    # Set up Score-CAM using the target layer from the model's encoder.
    target_layer = get_last_conv_layer(model)
    if target_layer is None:
        raise ValueError("Could not find a convolutional layer in the encoder")
    score_cam = ScoreCAM(model, target_layer)

    # Create output directory for results.
    output_path = os.path.join(current_dir, "results", f"{head_name}_ScoreCAM")
    os.makedirs(output_path, exist_ok=True)

    # Create CSV file for predictions.
    csv_file_path = os.path.join(output_path, 'all_predictions.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Bag ID', 'Image Index', 'True Label', 'Bag Prediction', 'Instance Prediction'])

        for (images, yb, instance_yb, bag_id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)):
            # Move tensors to GPU and clone if necessary.
            if not isinstance(images, list):
                images = images.clone().cuda()
                yb = yb.clone().cuda()
            else:
                images = [img.clone().cuda() for img in images]
                yb = yb.clone().cuda()
                
            bag_pred, _, instance_predictions, _ = model(images, pred_on=True)
            cams = score_cam(images)  # Get Score-CAMs for all images

            # Process each bag and instance image.
            for bag_index, bag in enumerate(images):
                for i in range(bag.size(0)):
                    img_tensor = bag[i].unsqueeze(0).to(device)
                    # Get the corresponding CAM for this image.
                    cam = cams[bag_index][i].squeeze().detach().cpu().numpy()
                    # Apply the heatmap overlay and save the result.
                    result = apply_heatmap(img_tensor, cam)
                    result_path = os.path.join(output_path, f'scorecam_bag{bag_id[bag_index]}_img{i}.png')
                    result.save(result_path)
                    # Write prediction details to CSV.
                    instance_pred = instance_predictions[bag_index][i]
                    csv_writer.writerow([bag_id[bag_index], i, yb[bag_index].item(), 
                                         bag_pred[bag_index].item(), instance_pred.item()])

    print(f"CSV file with all predictions has been saved to {csv_file_path}")

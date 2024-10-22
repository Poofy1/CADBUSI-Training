import os, sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from data.format_data import *
from data.sudo_labels import *
from loss.palm import PALM
from data.save_arch import *
from archs.model_PALM2_solo_saliency import *
from data.bag_loader import *
from data.instance_loader import *
from PIL import ImageDraw, ImageFont

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output, _, instance_predictions, _ = self.model([x], pred_on=True)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1

        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.detach()
        activations = self.activations.detach()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

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
    head_name = "Palm2_OFFICIAL_SAL_efficientnet_b0"
    model_version = "1" #Leave "" to read HEAD
    
    # loaded configuration
    model_path = os.path.join(model_folder, head_name, model_version)
    config = load_model_config(model_path)

    # Paths
    output_path = f"{current_dir}/results/{head_name}_Map/"
    mkdir(output_path, exist_ok=True)

    # Get Training Data
    bags_train, bags_val = prepare_all_data(config)
    num_labels = len(config['label_columns'])
    
    val_transform = T.Compose([
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # Create datasets
    dataset_val = BagOfImagesDataset(bags_val, transform=val_transform)
    val_dl = TUD.DataLoader(dataset_val, batch_size=1, collate_fn = collate_bag, drop_last=True)


    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes=num_labels).to(device)
    
    # Load the saved model state
    if model_version:
        model_path = f"{model_folder}/{head_name}/{model_version}/model.pth"
    else:
        model_path = f"{model_folder}/{head_name}/model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Setup Grad-CAM
    target_layer = model.encoder[-1][-1] # Last layer of encoder
    grad_cam = GradCAM(model, target_layer)

    # Create datasets and dataloaders
    val_transform = T.Compose([
        CLAHETransform(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_val = BagOfImagesDataset(bags_val, transform=val_transform)
    val_dl = DataLoader(dataset_val, batch_size=1, collate_fn=collate_bag, drop_last=True)

    # Create output directory
    output_path = f"{current_dir}/results/{head_name}_GradCAM/"
    os.makedirs(output_path, exist_ok=True)

    # Create CSV file for predictions
    csv_file_path = os.path.join(output_path, 'all_predictions.csv')

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Bag ID', 'Image Index', 'True Label', 'Bag Prediction', 'Instance Prediction'])

        for (data, yb, instance_yb, bag_id) in tqdm(val_dl, total=len(val_dl)):
            xb, yb = data, yb.to(device)
            bag_pred, bag_instance_predictions, instance_predictions, _ = model(xb, pred_on=True)

            for bag_index, bag in enumerate(xb):
                for i in range(bag.size(0)):
                    img_tensor = bag[i].unsqueeze(0).to(device)
                    
                    # Generate Grad-CAM
                    cam = grad_cam(img_tensor)
                    cam = cam.squeeze().cpu().numpy()

                    # Process the original image
                    img = TF.to_pil_image(unnormalize(img_tensor.squeeze().cpu()))
                    
                    # Create heatmap
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = Image.fromarray(heatmap).resize(img.size, Image.BILINEAR)

                    # Overlay heatmap on original image
                    result = Image.blend(img.convert('RGB'), heatmap, 0.5)

                    # Save the result
                    result_path = os.path.join(output_path, f'gradcam_bag{bag_id[bag_index]}_img{i}.png')
                    result.save(result_path)

                    # Write to CSV
                    instance_pred = instance_predictions[bag_index * bag.size(0) + i].item()
                    csv_writer.writerow([bag_id[bag_index], i, yb[bag_index].item(), bag_pred[bag_index].item(), instance_pred])

    print(f"CSV file with all predictions has been saved to {csv_file_path}")
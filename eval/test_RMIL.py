import torch
import os
import sys
import torch.utils.data as TUD
from tqdm import tqdm
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from util.eval_util import *
from data.format_data import *
from data.sudo_labels import *
from data.save_arch import *
from archs.model_INS import *
from data.bag_loader import *
from data.instance_loader import *
from config import *
from loss.IWSCL import *




def test_model(model, bag_dataloader, instance_dataloader):

    # Bag-level metrics
    bag_targets, bag_predictions = [], []
    
    # Instance-level metrics
    instance_targets, fc_predictions = [], []
    instance_info = []
    instance_features = []
    
    model.eval()
    with torch.no_grad():
        # Bag-level testing
        for images, yb, _, _ in tqdm(bag_dataloader, desc="Testing bags"):
            bag_pred, _, instance_pred, _, _, _ = model(images, bag_on=True)
            bag_targets.extend(yb.cpu().numpy())
            bag_predictions.extend((bag_pred > 0.5).float().cpu().numpy())
        
        for idx, ((im_q, im_k), instance_labels, unique_id) in enumerate(tqdm(instance_dataloader, total=len(instance_dataloader))):
            im_q = im_q.cuda(non_blocking=True)
            im_k = im_k.cuda(non_blocking=True)
            instance_labels = instance_labels.cuda(non_blocking=True)
            _, _, fc_pred, features, iwscl_loss, pseudo_labels = model(im_q, im_k, true_label = instance_labels, projector=True)
            
            instance_info.extend(unique_id) 
            instance_features.extend(features.cpu().numpy())
            
            instance_targets.extend(instance_labels.cpu().numpy())
            # Check if fc_pred is None and handle accordingly
            if fc_pred is None:
                fc_predictions.extend([0] * len(instance_labels))
            else:
                fc_predictions.extend((fc_pred > 0.5).float().cpu().numpy())
                
    return (np.array(bag_targets), np.array(bag_predictions), 
            np.array(instance_targets), np.array(fc_predictions),
            instance_info, np.array(instance_features))





if __name__ == '__main__':
    # Get the parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = f'{current_dir}/results/RMIL_OOD/'
    os.makedirs(output_path, exist_ok=True)
    
    # Config
    model_version = '3'
    head_name = "CADBUSI_R-MIL_64"
    data_config = LesionDataConfig #FishDataConfig or LesionDataConfig
    
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create bag datasets
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
    #bag_dataset_train = SyntheticBagDataset(bags_train, transform=train_transform)
    bag_dataset_val = BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False)
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=config['bag_batch_size'], collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_size=config['bag_batch_size'], collate_fn = collate_bag, drop_last=True)

    instance_dataloader_train = Instance_Dataset(bags_train, [], transform=val_transform, warmup=False, dual_output=True)
    instance_dataloader_train = TUD.DataLoader(instance_dataloader_train, batch_size=config['instance_batch_size'], collate_fn=collate_instance, shuffle=False)
    instance_dataset_test = Instance_Dataset(bags_val, [], transform=val_transform, warmup=True, dual_output=True)
    instance_dataloader_test = TUD.DataLoader(instance_dataset_test, batch_size=config['instance_batch_size'], collate_fn=collate_instance, shuffle=False)

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    model, optimizer, state = setup_model(model, config)

    # Test the model
    results = test_model(model, bag_dataloader_val, instance_dataloader_train)
    bag_targets, bag_predictions, instance_targets, fc_predictions, instance_info, instance_features = results
    
    
    
    # Get the model and access IWSCL
    iwscl = model.iwscl
    prototypes = iwscl.prototypes
    proto_class_counts = iwscl.proto_class_counts  # Shape: [num_classes, num_classes]
    prototype_labels = torch.argmax(proto_class_counts, dim=1)  # Shape: [num_classes]

    # Visualize prototypes and instances
    visualize_prototypes_and_instances(prototypes, prototype_labels, instance_features, instance_targets, config['dataset_name'], config['head_name'], output_path)
    

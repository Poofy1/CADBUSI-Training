import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from archs.save_arch import *
from torch.optim import Adam
from data.format_data import *
from archs.model_ABMIL import *
from archs.backbone import create_timm_body
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def collate_custom(batch):
    batch_data = []
    batch_bag_labels = []
    batch_instance_labels = []
    batch_ids = []  # List to store bag IDs

    for sample in batch:
        image_data, bag_labels, instance_labels, bag_id = sample  # Updated to unpack four items
        batch_data.append(image_data)
        batch_bag_labels.append(bag_labels)
        batch_instance_labels.append(instance_labels)
        batch_ids.append(bag_id)

    # Use torch.stack for bag labels to handle multiple labels per bag
    out_bag_labels = torch.stack(batch_bag_labels).cuda()

    # Converting to a tensor
    out_ids = torch.tensor(batch_ids, dtype=torch.long).cuda()

    return batch_data, out_bag_labels, batch_instance_labels, out_ids



class EmbeddingBagModel(nn.Module):
    
    def __init__(self, encoder, aggregator, num_classes=1):
        super(EmbeddingBagModel,self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.num_classes = num_classes

    def forward(self, input):
        num_bags = len(input) # input = [bag #, image #, channel, height, width]
        
        # Concatenate all bags into a single tensor for batch processing
        all_images = torch.cat(input, dim=0)  # Shape: [Total images in all bags, channel, height, width]
        
        # Calculate the embeddings for all images in one go
        h_all = self.encoder(all_images.cuda())
        
        # Split the embeddings back into per-bag embeddings
        split_sizes = [bag.size(0) for bag in input]
        h_per_bag = torch.split(h_all, split_sizes, dim=0)
        logits = torch.empty(num_bags, self.num_classes).cuda()
        saliency_maps, yhat_instances, attention_scores = [], [], []
        
        for i, h in enumerate(h_per_bag):
            # Receive four values from the aggregator
            yhat_bag, sm, yhat_ins, att_sc = self.aggregator(h)
            
            logits[i] = yhat_bag
            saliency_maps.append(sm)
            yhat_instances.append(yhat_ins)
            attention_scores.append(att_sc)
        
        return logits, saliency_maps, yhat_instances, attention_scores


if __name__ == '__main__':

    # Config
    model_name = 'imagenette-ABMIL'
    
    
    encoder_arch = 'resnet18'
    dataset_name = 'cifar10'
    label_columns = ['Has_Truck']
    instance_columns = ['']
    #dataset_name = 'export_01_31_2024'
    #label_columns = ['Has_Malignant']
    #instance_columns = ['Reject Image', 'Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 32
    batch_size = 30
    min_bag_size = 2
    max_bag_size = 25
    epochs = 500
    lr = 0.001


    dataset_name = 'imagenette2'
    label_columns = ['Has_Fish']
    instance_columns = ['Has_Fish']  
    img_size = 128
    min_bag_size = 2
    max_bag_size = 25


    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    #export_location = '/home/paperspace/cadbusi-LFS/export_09_28_2023/'
    #cropped_images = f"/home/paperspace/Temp_Data/{img_size}_images/"

    # Get Training Data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)
    num_labels = len(label_columns)
    
    train_transform = T.Compose([
                ###T.RandomVerticalFlip(),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.RandomAffine(degrees=(-45, 45), translate=(0.05, 0.05), scale=(1, 1.2),),
                CLAHETransform(),
                T.ToTensor(),
                ###GaussianNoise(mean=0, std=0.015), 
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    val_transform = T.Compose([
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    
    # Create datasets
    #dataset_train = TUD.Subset(BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False),list(range(0,100)))
    #dataset_val = TUD.Subset(BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False),list(range(0,100)))
    dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
    dataset_val = BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False)

    
                    
    # Create data loaders
    train_dl =  TUD.DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)

    encoder = create_timm_body(encoder_arch, pretrained=False)
    nf = num_features_model( nn.Sequential(*encoder.children()))
    
    # bag aggregator
    #aggregator = ABMIL_aggregate( nf = nf, num_classes = num_labels, pool_patches = 6, L = 128)
    aggregator = Linear_Classifier(nf= nf, num_classes = num_labels)
    
    # total model
    bagmodel = EmbeddingBagModel(encoder, aggregator, num_classes = num_labels).cuda()
    total_params = sum(p.numel() for p in bagmodel.parameters())
    print(f"Total Parameters: {total_params}")
        
        
    optimizer = Adam(bagmodel.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    epoch_start = 0
    
    # Check if the model already exists
    model_folder = f"{env}/models/{model_name}/"
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"
    
    if os.path.exists(model_path):
        bagmodel.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optimizer_path))
        print(f"Loaded pre-existing model from {model_name}")
        
        with open(stats_path, 'rb') as f:
            saved_stats = pickle.load(f)
            train_losses_over_epochs = saved_stats['train_losses']
            valid_losses_over_epochs = saved_stats['valid_losses']
            epoch_start = saved_stats['epoch']
            val_loss_best = saved_stats['val_loss']
    else:
        print(f"{model_name} does not exist, creating new instance")
        os.makedirs(model_folder, exist_ok=True)
        val_loss_best = 99999
    
    
    print("Training Data...")
    # Training loop
    for epoch in range(epoch_start, epochs):
        # Training phase
        bagmodel.train()
        total_loss = 0.0
        total_acc = 0
        total = 0
        correct = [0] * num_labels
        for (data, yb, instance_yb, id) in tqdm(train_dl, total=len(train_dl)): 
            xb, yb = data, yb.cuda()
            
            optimizer.zero_grad()
            
            outputs, _, _, _ = bagmodel(xb)

            loss = loss_func(outputs, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            predicted = (outputs > .5).float()
            total += yb.size(0)
            
            for label_idx in range(num_labels):
                correct[label_idx] += (predicted[:, label_idx] == yb[:, label_idx]).sum().item()
            
        train_loss = total_loss / total
        train_acc = [total_correct / total for total_correct in correct]


        # Evaluation phase
        bagmodel.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        total = 0
        correct = [0] * num_labels
        all_targs = []
        all_preds = []
        with torch.no_grad():
            for (data, yb, instance_yb, id) in tqdm(val_dl, total=len(val_dl)): 
                xb, yb = data, yb.cuda()

                outputs, _, _, _ = bagmodel(xb)
                
                # TEMP DEBUG
                """if not torch.all(outputs.ge(0) & outputs.le(1)):
                    print(f"Invalid output detected: {outputs[~(outputs.ge(0) & outputs.le(1))]}")
                if not torch.all(yb.ge(0) & yb.le(1)):
                    print(f"Invalid label detected: {yb[~(yb.ge(0) & yb.le(1))]}")"""
                
                loss = loss_func(outputs, yb)
                
                total_val_loss += loss.item() * len(xb)
                predicted = (outputs > .5).float()
                total += yb.size(0)
                
                for label_idx in range(num_labels):
                    correct[label_idx] += (predicted[:, label_idx] == yb[:, label_idx]).sum().item()
                
                # Confusion Matrix data
                all_targs.extend(yb.cpu().numpy())
                if len(predicted.size()) == 0:
                    predicted = predicted.view(1)
                all_preds.extend(predicted.cpu().detach().numpy())

        val_loss = total_val_loss / total
        val_acc = [total_correct / total for total_correct in correct]
        
        train_losses_over_epochs.append(train_loss)
        valid_losses_over_epochs.append(val_loss)
        
        # Constructing header with label names
        acc_headers = " | ".join(f"Acc ({name})" for name in label_columns)
        header = f"Epoch {epoch+1} | {acc_headers} | Loss"

        # Constructing training and validation accuracy strings
        train_acc_str = " | ".join(f"{acc:.4f}" for acc in train_acc)
        val_acc_str = " | ".join(f"{acc:.4f}" for acc in val_acc)

        # Printing epoch summary
        print(header)
        print(f"Train   | {train_acc_str} | {train_loss:.4f}")
        print(f"Val     | {val_acc_str} | {val_loss:.4f}")
        
        
        # Save the model
        if val_loss < val_loss_best:
            val_loss_best = val_loss  # Update the best validation accuracy
            save_state(epoch, label_columns, train_acc, val_loss, val_acc, model_folder, model_name, bagmodel, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs)
            print("Saved checkpoint due to improved val_loss")
import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from archs.save_arch import *
from torch.optim import Adam
from data.format_data import *
from archs.model_FC import *
from archs.backbone import create_timm_body
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


    
    
def collate_custom(batch):
    batch_data = []
    batch_labels = []
    batch_ids = []  # List to store bag IDs

    for sample in batch:
        image_data, labels, bag_id = sample
        batch_data.append(image_data)
        batch_labels.append(labels)  # labels are already tensors
        batch_ids.append(bag_id)  # Append the bag ID

    # Using torch.stack for labels to handle multiple labels per bag
    out_labels = torch.stack(batch_labels).cuda()

    # Converting to a tensor
    out_ids = torch.tensor(batch_ids, dtype=torch.long).cuda()

    return batch_data, out_labels, out_ids


class EmbeddingBagModel(nn.Module):
    
    def __init__(self, encoder, aggregator, num_classes=1):
        super(EmbeddingBagModel,self).__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.num_classes = num_classes

    def forward(self, input):
        num_bags = len(input)  # input = [bag #, image #, channel, height, width]
        
        # Concatenate all bags into a single tensor for batch processing
        all_images = torch.cat(input, dim=0)  # Shape: [Total images in all bags, channel, height, width]
        
        # Calculate the embeddings for all images in one go
        h_all = self.encoder(all_images)
        
        # Split the embeddings back into per-bag embeddings
        split_sizes = [bag.size(0) for bag in input]
        h_per_bag = torch.split(h_all, split_sizes, dim=0)
        logits = torch.empty(num_bags, self.num_classes).cuda()
        yhat_instances, attention_scores = [], []
        
        for i, h in enumerate(h_per_bag):
            # Receive values from the aggregator
            yhat_bag, yhat_ins, att_sc = self.aggregator(h)
            
            logits[i] = yhat_bag
            yhat_instances.append(yhat_ins)
            attention_scores.append(att_sc)
        
        return logits, yhat_instances, attention_scores


if __name__ == '__main__':

    # Config
    model_name = 'test'
    encoder_arch = 'resnet18'
    dataset_name = 'export_11_11_2023'
    label_columns = ['Has_Malignant', 'Has_Benign']
    img_size = 350
    batch_size = 5
    min_bag_size = 2
    max_bag_size = 20
    epochs = 500
    lr = 0.001

    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    #export_location = '/home/paperspace/cadbusi-LFS/export_09_28_2023/'
    #cropped_images = f"/home/paperspace/Temp_Data/{img_size}_images/"
    
    # Get Training Data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, cropped_images, img_size, min_bag_size, max_bag_size)


    print("Training Data...")
    # Create datasets
    #dataset_train = TUD.Subset(BagOfImagesDataset(bags_train, save_processed=False),list(range(0,100)))
    #dataset_val = TUD.Subset(BagOfImagesDataset(bags_val, save_processed=False),list(range(0,100)))
    dataset_train = BagOfImagesDataset(bags_train, save_processed=False)
    dataset_val = BagOfImagesDataset(bags_val, train=False)
            
    # Create data loaders
    train_dl =  TUD.DataLoader(dataset_train, batch_size=batch_size, collate_fn = collate_custom, drop_last=True, shuffle = True)
    val_dl =    TUD.DataLoader(dataset_val, batch_size=batch_size, collate_fn = collate_custom, drop_last=True)

    
    # Check if the model already exists
    model_folder = f"{env}/models/{model_name}/"
    model_path = f"{model_folder}/{model_name}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"

    encoder = create_timm_body(encoder_arch)
    nf = num_features_model( nn.Sequential(*encoder.children()))
    
    # bag aggregator
    aggregator = FC_aggregate( nf = nf, num_classes = len(label_columns), L = 128, fc_layers=[256, 64], dropout = .6)

    # total model
    bagmodel = EmbeddingBagModel(encoder, aggregator, num_classes = len(label_columns)).cuda()
    total_params = sum(p.numel() for p in bagmodel.parameters())
    print(f"Total Parameters: {total_params}")
        
        
    optimizer = Adam(bagmodel.parameters(), lr=lr)
    
    # Use BCE since we are doing multi label instead of multi class
    loss_func = nn.BCELoss()

    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    num_labels = len(label_columns)
    epoch_start = 0
    
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
    

    # Training loop
    for epoch in range(epoch_start, epochs):
        # Training phase
        bagmodel.train()
        total_loss = 0.0
        total_acc = 0
        total = 0
        correct = [0] * num_labels
        for (data, yb, _) in tqdm(train_dl, total=len(train_dl)): 
            xb, yb = data, yb.cuda()
            
            # Train
            optimizer.zero_grad()
            outputs, _, _ = bagmodel(xb)
            loss = loss_func(outputs, yb)
            loss.backward()
            optimizer.step()

            # Accuracy 
            total_loss += loss.item() * len(xb)
            predicted = (outputs > .5).float()
            total += yb.size(0)
            
            # Calculate correct predictions for each label
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
            for (data, yb, _) in tqdm(val_dl, total=len(val_dl)): 
                xb, yb = data, yb.cuda()

                outputs, _, _ = bagmodel(xb)
                loss = loss_func(outputs, yb)
                
                total_val_loss += loss.item() * len(xb)
                
                total_loss += loss.item() * len(xb)
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
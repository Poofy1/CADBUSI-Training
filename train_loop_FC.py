import os, pickle
from timm import create_model
from fastai.vision.all import *
import torch.utils.data as TUD
from fastai.vision.learner import _update_first_layer
from tqdm import tqdm
from torch import nn
from training_eval import *
from torch.optim import Adam
from data_prep import *
from model_FC import *
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# this function is used to cut off the head of a pretrained timm model and return the body
def create_timm_body(arch:str, pretrained=True, cut=None, n_in=3):
    "Creates a body from any model in the `timm` library."
    model = create_model(arch, pretrained=pretrained, num_classes=0, global_pool='')
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or function")

def create_custom_body(model, pretrained=True, cut=None, n_in=3):
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NameError("cut must be either integer or function")
    
    
    
def collate_custom(batch):
    batch_data = []
    batch_labels = []
    batch_ids = []  # List to store bag IDs

    for sample in batch:
        image_data, label, bag_id = sample
        batch_data.append(image_data)
        batch_labels.append(label)
        batch_ids.append(bag_id)  # Append the bag ID

    out_labels = torch.tensor(batch_labels).cuda()
    out_ids = torch.tensor(batch_ids).cuda()  # Convert bag IDs to a tensor
    
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
        
        print(h_per_bag.shape)
        
        for i, h in enumerate(h_per_bag):
            print(h.shape)
            # Receive values from the aggregator
            yhat_bag, yhat_ins, att_sc = self.aggregator(h)
            
            logits[i] = yhat_bag
            yhat_instances.append(yhat_ins)
            attention_scores.append(att_sc)
        
        return logits.squeeze(1), yhat_instances, attention_scores


if __name__ == '__main__':

    # Config
    pretrained_head = None
    model_name = 'test'
    encoder_arch = 'resnet18'
    img_size = 350
    batch_size = 5
    min_bag_size = 2
    max_bag_size = 20
    epochs = 500
    lr = 0.001

    # Paths
    export_location = 'D:/DATA/CASBUSI/exports/export_11_11_2023/'
    cropped_images = f"F:/Temp_SSD_Data/{img_size}_images/"
    #export_location = '/home/paperspace/cadbusi-LFS/export_09_28_2023/'
    #cropped_images = f"/home/paperspace/Temp_Data/{img_size}_images/"
    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
    image_data = pd.read_csv(f'{export_location}/ImageData.csv')
    
    
    bags_train, bags_val = prepare_all_data(export_location, case_study_data, breast_data, image_data, cropped_images, img_size, min_bag_size, max_bag_size)


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
    pretrained_path = f"{env}/models/{pretrained_head}/{pretrained_head}.pth"
    optimizer_path = f"{model_folder}/{model_name}_optimizer.pth"
    stats_path = f"{model_folder}/{model_name}_stats.pkl"
    
    model_exists = os.path.exists(model_path)
    
    encoder = create_timm_body(encoder_arch)
    nf = num_features_model( nn.Sequential(*encoder.children()))
    
    # bag aggregator
    aggregator = FC_aggregate( nf = nf, num_classes = 1, L = 128, fc_layers=[256, 64], dropout = .5)

    # total model
    bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
    total_params = sum(p.numel() for p in bagmodel.parameters())
    print(f"Total Parameters: {total_params}")
        
        
    optimizer = Adam(bagmodel.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
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
            val_acc_best = saved_stats.get('val_acc', -1)  # If 'val_acc' does not exist, default to -1
    else:
        print(f"{model_name} does not exist, creating new instance")
        os.makedirs(model_folder, exist_ok=True)
        val_acc_best = -1 
        
        """if pretrained_head is not None and os.path.exists(pretrained_path):
            print(f"Using pretrained head: {pretrained_head}")
            #bagmodel.load_state_dict(torch.load(pretrained_path))
            #encoder = create_custom_body(bagmodel)
            bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()"""
    
    
    
    # Training loop
    for epoch in range(epoch_start, epochs):
        # Training phase
        bagmodel.train()
        total_loss = 0.0
        total_acc = 0
        total = 0
        correct = 0
        for (data, yb, _) in tqdm(train_dl, total=len(train_dl)): 
            xb, yb = data, yb.cuda()
            
            optimizer.zero_grad()
            
            outputs, _, _ = bagmodel(xb)

            loss = loss_func(outputs, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            predicted = torch.round(outputs).squeeze()
            total += yb.size(0)
            correct += predicted.eq(yb.squeeze()).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total


        # Evaluation phase
        bagmodel.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        total = 0
        correct = 0
        all_targs = []
        all_preds = []
        with torch.no_grad():
            for (data, yb, _) in tqdm(val_dl, total=len(val_dl)): 
                xb, yb = data, yb.cuda()

                outputs, _, _ = bagmodel(xb)
                loss = loss_func(outputs, yb)
                
                total_val_loss += loss.item() * len(xb)
                predicted = torch.round(outputs).squeeze() 
                total += yb.size(0)
                correct += predicted.eq(yb.squeeze()).sum().item()
                
                # Confusion Matrix data
                all_targs.extend(yb.cpu().numpy())
                if len(predicted.size()) == 0:
                    predicted = predicted.view(1)
                all_preds.extend(predicted.cpu().detach().numpy())

        val_loss = total_val_loss / total
        val_acc = correct / total
        
        train_losses_over_epochs.append(train_loss)
        valid_losses_over_epochs.append(val_loss)
        
        print(f"Epoch {epoch+1} | Acc   | Loss")
        print(f"Train   | {train_acc:.4f} | {train_loss:.4f}")
        print(f"Val     | {val_acc:.4f} | {val_loss:.4f}")
        
        # Save the model
        if val_acc > val_acc_best:
            val_acc_best = val_acc  # Update the best validation accuracy
            save_state(epoch, train_acc, val_acc, model_folder, model_name, bagmodel, optimizer, all_targs, all_preds, train_losses_over_epochs, valid_losses_over_epochs)
            print("Saved checkpoint due to improved val_acc")
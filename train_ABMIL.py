import os
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from torch.optim import Adam
from data.format_data import *
from archs.model_ABMIL import *
from data.bag_loader import *
from config import *
env = os.path.dirname(os.path.abspath(__file__))
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "ABMIL_OFFICAL"
    data_config = FishDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])
    
    train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                T.RandomAffine(degrees=(-90, 90), translate=(0.05, 0.05), scale=(1, 1.2),),
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    val_transform = T.Compose([
                CLAHETransform(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    
    # Create bag datasets
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
    bag_dataset_val = BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False)
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=config['bag_batch_size'], collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_size=config['bag_batch_size'], collate_fn = collate_bag, drop_last=True)

    
    # total model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}") 
        
        
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    loss_func = nn.BCELoss()
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, optimizer, config)
    
    
    print("Training Data...")
    # Training loop
    while state['epoch'] < config['total_epochs']:
        # Training phase
        model.train()
        total_loss = 0.0
        total_acc = 0
        total = 0
        correct = [0] * num_labels
        for (data, yb, instance_yb, id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)): 
            xb, yb = data, yb.cuda()
            
            optimizer.zero_grad()
            
            outputs, _, _, _ = model(xb)

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
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        total = 0
        correct = [0] * num_labels
        all_targs = []
        all_preds = []
        with torch.no_grad():
            for (data, yb, instance_yb, id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                xb, yb = data, yb.cuda()

                outputs, _, _, _ = model(xb)

                
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
        acc_headers = " | ".join(f"Acc ({name})" for name in config['label_columns'])
        header = f"Epoch {state['epoch']+1} | {acc_headers} | Loss"

        # Constructing training and validation accuracy strings
        train_acc_str = " | ".join(f"{acc:.4f}" for acc in train_acc)
        val_acc_str = " | ".join(f"{acc:.4f}" for acc in val_acc)

        # Printing epoch summary
        print(header)
        print(f"Train   | {train_acc_str} | {train_loss:.4f}")
        print(f"Val     | {val_acc_str} | {val_loss:.4f}")
        
        target_folder = state['head_folder']
        target_name = state['pretrained_name']
        
        # Save the model
        if val_loss < state['val_loss_bag']:
            state['val_loss_bag'] = val_loss  # Update the best validation accuracy
            save_state(state, config, train_acc, val_loss, val_acc, model, optimizer)
            print("Saved checkpoint due to improved val_loss")
import os, pickle
from fastai.vision.all import *
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from archs.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from util.format_data import *
from util.sudo_labels import *
from archs.model_PALM2_solo import *
from archs.model_INS import *
from data.bag_loader import *
from data.instance_loader_RMIL import *
from loss.R_MIL import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == '__main__':

    # Config
    model_version = '2'
    head_name = "TESTING"
    
    """dataset_name = 'export_oneLesions' #'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']  
    img_size = 224
    bag_batch_size = 3
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    arch = 'efficientnet_b0'
    pretrained_arch = False"""

    
    dataset_name = 'imagenette2_hard'
    label_columns = ['Has_Fish']
    instance_columns = ['Has_Fish']  
    img_size = 128
    bag_batch_size = 5
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  25
    arch = 'efficientnet_b0'
    pretrained_arch = False

    #ITS2CLR Config
    feature_extractor_train_count = 3
    MIL_train_count = 3
    initial_ratio = .3 #0.3 # --% preditions included
    final_ratio = .8 #0.85 # --% preditions included
    total_epochs = 100
    warmup_epochs = 3
    learning_rate=0.001
    reset_aggregator = True # Reset the model.aggregator weights after contrastive learning


    mix_alpha=0.2  #0.2
    mix='mixup'
    
    
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

    
    # Get Training Data
    config = {
        "head_name": head_name,
        "model_version": model_version,
        "dataset_name": dataset_name,
        "arch": arch,
        "pretrained_arch": pretrained_arch,
        "label_columns": label_columns,
        "instance_columns": instance_columns,
        "img_size": img_size,
        "bag_batch_size": bag_batch_size,
        "min_bag_size": min_bag_size,
        "max_bag_size": max_bag_size,
        "instance_batch_size": instance_batch_size,
        "feature_extractor_train_count": feature_extractor_train_count,
        "MIL_train_count": MIL_train_count,
        "initial_ratio": initial_ratio,
        "final_ratio": final_ratio,
        "total_epochs": total_epochs,
        "reset_aggregator": reset_aggregator,
        "warmup_epochs": warmup_epochs,
        "learning_rate": learning_rate,
    }
    bags_train, bags_val = prepare_all_data(config)
    num_classes = len(label_columns) + 1
    num_labels = len(label_columns)

    # Create bag datasets
    bag_dataset_train = BagOfImagesDataset(bags_train, transform=train_transform, save_processed=False)
    bag_dataset_val = BagOfImagesDataset(bags_val, transform=val_transform, save_processed=False)
    bag_dataloader_train = TUD.DataLoader(bag_dataset_train, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True, shuffle = True)
    bag_dataloader_val = TUD.DataLoader(bag_dataset_val, batch_size=bag_batch_size, collate_fn = collate_bag, drop_last=True)

    # Create Instance datasets
    instance_dataset_train = Instance_Dataset(bags_train, transform=train_transform, known_only = False, pos_only = False)
    instance_dataset_val = Instance_Dataset(bags_val, transform=val_transform, known_only = False, pos_only = False)
    train_sampler = BalancedSampler(instance_dataset_train, batch_size=instance_batch_size)
    val_sampler = BalancedSampler(instance_dataset_val, batch_size=instance_batch_size)
    instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=1, collate_fn = collate_instance, pin_memory=True)
    instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
    
    
    
    # Create Model
    class Args:
        def __init__(self):
            # Dataset
            self.dataset = 'cifar10'  # or 'cub200', etc.
            self.num_class = 2  # number of classes in the dataset
            
            # Model architecture
            self.arch = 'resnet18'  # base encoder architecture
            self.low_dim = 128  # dimension of embeddings
            
            # MoCo specific configs
            self.moco_queue = 8000  # queue size
            self.moco_m = 0.999  # moco momentum of updating key encoder
            
            # Prototype learning
            self.proto_m = 0.99  # momentum for updating prototypes
            
            self.epochs = 10
            self.conf_ema_range= [0.95, 0.8]
            
    args = Args()
    model = INS(args, SupConResNet).cuda()
    #model = Embeddingmodel(arch, pretrained_arch, num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    
    
    BCE_loss = nn.BCEWithLogitsLoss() # for testing
    
    loss_fn = partial_loss(instance_dataset_train.partialY.cuda())
    loss_fn_pos = partial_loss(instance_dataset_train.partialY_pos.cuda())
    cls_loss = Cls_loss(instance_dataset_train.partialY_cls.cuda())
    loss_cont_fn = SupConLoss()
    CE_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001)
    

    model, optimizer, state = setup_model(model, optimizer, config)
    
    # Training loop
    while state['epoch'] < total_epochs:
        
        loss_fn.set_conf_ema_m(state['epoch'], args) #Setup loss
        
        if not state['pickup_warmup']: # Are we resuming from a head model?
        
            
            
            if state['warmup']:
                target_count = warmup_epochs
            else:
                target_count = feature_extractor_train_count
            
            
            

            print('Training Feature Extractor')
            print(f'Warmup Mode: {state["warmup"]}')

            
            for iteration in range(target_count): 
                model.train()
                losses = AverageMeter()
                palm_total_correct = 0
                instance_correct  = 0
                total_samples = 0
                
                # Iterate over the training data
                for idx, (im_q, im_k, instance_labels, bag_label, point, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                    im_q = im_q.cuda(non_blocking=True)
                    im_k = im_k.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)
                    bsz = instance_labels.shape[0]

                    # forward
                    optimizer.zero_grad()
                    instance_predictions, features_cont, pseudo_score_cont, partial_target_cont, score_prot = model(im_q, im_k, partial_Y=instance_labels, args=args)
                    print(instance_predictions)
                    print(pseudo_score_cont)
                    # BCE loss (assuming your model now outputs 2 values per instance)
                    #bce_loss_value = BCE_loss(instance_predictions[:bsz], instance_labels)
                    
                    if not state['warmup']:
                        pseudo_target_max, pseudo_target_cont = torch.max(pseudo_score_cont, dim=1)  # 8194,
                        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)  # 8194,1 
        
                        loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=unique_id, batchY=instance_labels, point=point)
                        mask = torch.eq(pseudo_target_cont[:bsz], pseudo_target_cont.T).float().cuda()
                    else: 
                        mask = None
                        
                    # contrastive loss
                    loss_cont = loss_cont_fn(features=features_cont, mask=mask, batch_size=bsz)
                    # classification loss 
                    loss_cls = loss_fn(instance_predictions, unique_id)
        
                    # Backward pass and optimization step
                    total_loss = loss_cls + loss_cont * .0
                    total_loss.backward()
                    optimizer.step()

                    # Update the loss meter
                    losses.update(total_loss.item(), bsz)       
                    
                    # Assuming instance_predictions has shape [batch_size, 2]
                    predicted = (torch.sigmoid(instance_predictions[:, 1]) > 0.5).float()
                    instance_correct += (predicted == instance_labels[:, 1]).sum().item()  # Compare with the second number in instance_labels
                    total_samples += instance_labels.size(0) 
                
                instance_train_acc = instance_correct / total_samples
                
                # Validation loop
                model.eval()
                palm_total_correct = 0
                instance_correct = 0
                total_samples = 0
                val_losses = AverageMeter()

                with torch.no_grad():
                    for idx, (im_q, im_k, instance_labels, bag_label, point, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                        im_q = im_q.cuda(non_blocking=True)
                        im_k = im_k.cuda(non_blocking=True)
                        instance_labels = instance_labels.cuda(non_blocking=True)
                        bsz = instance_labels.shape[0]

                        # Forward pass
                        instance_predictions, features_cont, pseudo_score_cont, partial_target_cont, score_prot = model(im_q, im_k, partial_Y=instance_labels, args=args)

                        # Calculate total loss
                        #bce_loss_value = BCE_loss(instance_predictions[:bsz], instance_labels)
                        
                        # contrastive loss
                        loss_cont = loss_cont_fn(features=features_cont, mask=None, batch_size=bsz)
                        # classification loss 
                        loss_cls = loss_fn(instance_predictions, unique_id)
                        
                        total_loss = loss_cls + loss_cont * .5
                        
                        # Update the loss meter
                        val_losses.update(total_loss.item(), bsz)
                        
                        # Assuming instance_predictions has shape [batch_size, 2]
                        predicted = (torch.sigmoid(instance_predictions[:, 1]) > 0.5).float()
                        instance_correct += (predicted == instance_labels[:, 1]).sum().item()  # Compare with the second number in instance_labels
                        total_samples += instance_labels.size(0)

                
                instance_val_acc = instance_correct / total_samples
                
                print(f'[{iteration+1}/{target_count}] Train Loss: {losses.avg:.5f},  Train Acc: {instance_train_acc:.5f}')
                print(f'[{iteration+1}/{target_count}] Val Loss:   {val_losses.avg:.5f}, Val Acc: {instance_val_acc:.5f}')
                
                
                # Save the model
                if val_losses.avg < state['val_loss_instance']:
                    state['val_loss_instance'] = val_losses.avg
                    if state['warmup']:
                        target_folder = state['head_folder']
                        target_name = state['pretrained_name']
                    else:
                        target_folder = state['model_folder']
                        target_name = state['model_name']
                    all_targs = []
                    all_preds = []
                    
                    
                    if state['warmup']:
                        save_state(state['epoch'], label_columns, instance_train_acc, val_losses.avg, instance_val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                        print("Saved checkpoint due to improved val_loss_instance")




        if state['pickup_warmup']: 
            state['pickup_warmup'] = False
        if state['warmup']:
            print("Warmup Phase Finished")
            state['warmup'] = False
        
            
        print('\nTraining Bag Aggregator')
        for iteration in range(MIL_train_count):
            model.train()
            train_bag_logits = {}
            total_loss = 0.0
            total_acc = 0
            total = 0
            correct = 0

            for (images, yb, instance_labels, id) in tqdm(bag_dataloader_train, total=len(bag_dataloader_train)):
                optimizer.zero_grad()
                
                total_bag_loss = 0
                bag_preds = []
                
                # Process each bag separately
                for i, bag in enumerate(images):

                    # Forward pass for a single bag
                    bag_pred = model(bag.cuda(), args, bag_flag=True)
                    
                    # Move yb to CUDA and convert to one-hot encoding
                    yb_cuda = yb[i].unsqueeze(0).cuda()
                    yb_one_hot = F.one_hot(yb_cuda.long(), num_classes=2).float().squeeze(0)
                    
                    # Calculate loss for this bag
                    bag_loss = CE_loss(bag_pred, yb_one_hot)
                    
                    # Accumulate the loss
                    total_bag_loss += bag_loss
                    
                    # Store the prediction
                    bag_preds.append(bag_pred)
                    
                # Backward pass with accumulated loss
                total_bag_loss.backward()
                
                # Optimizer step
                optimizer.step()
                
                # Concatenate all bag predictions
                bag_preds = torch.cat(bag_preds, dim=0)
                
                
                # Update metrics
                total_loss += total_bag_loss.item()
                predicted = torch.argmax(bag_preds, dim=1)
                total += yb.size(0)
                yb_squeezed = yb.squeeze()  # This removes the extra dimension
                correct += (predicted.cpu() == yb_squeezed.cpu()).sum().item()

            
            train_loss = total_loss / total
            train_acc = correct / total
                    
                    
            # Evaluation phase
            model.eval()
            total = 0
            correct = 0
            total_val_loss = 0.0
            total_val_acc = 0.0
            all_targs = []
            all_preds = []

            with torch.no_grad():
                for (images, yb, instance_labels, id) in tqdm(bag_dataloader_val, total=len(bag_dataloader_val)): 
                    total_bag_loss = 0
                    bag_preds = []
                    
                    # Process each bag separately
                    for i, bag in enumerate(images):

                        # Forward pass for a single bag
                        bag_pred = model(bag.cuda(), args, bag_flag=True)
                        
                        # Move yb to CUDA and convert to one-hot encoding
                        yb_cuda = yb[i].unsqueeze(0).cuda()
                        yb_one_hot = F.one_hot(yb_cuda.long(), num_classes=2).float().squeeze(0)
                        
                        # Calculate loss for this bag
                        bag_loss = BCE_loss(bag_pred, yb_one_hot)
                        
                        # Accumulate the loss
                        total_bag_loss += bag_loss
                        
                        # Store the prediction
                        bag_preds.append(bag_pred)
                    
                    # Concatenate all bag predictions
                    bag_preds = torch.cat(bag_preds, dim=0)
                    
                    # Update metrics
                    total_val_loss += total_bag_loss.item()
                    predicted = torch.argmax(bag_preds, dim=1)
                    total += yb.size(0)
                    yb_squeezed = yb.squeeze()  # This removes the extra dimension
                    correct += (predicted.cpu() == yb_squeezed.cpu()).sum().item()
                        
            val_loss = total_val_loss / total
            val_acc = correct / total
                
            state['train_losses'].append(train_loss)
            state['valid_losses'].append(val_loss)    
            
            print(f"[{iteration+1}/{MIL_train_count}] | Acc | Loss")
            print(f"Train | {train_acc:.4f} | {train_loss:.4f}")
            print(f"Val | {val_acc:.4f} | {val_loss:.4f}")

            # Save the model
            if val_loss < state['val_loss_bag']:
                state['val_loss_bag'] = val_loss
                if state['warmup']:
                    target_folder = state['head_folder']
                    target_name = state['pretrained_name']
                else:
                    target_folder = state['model_folder']
                    target_name = state['model_name']
                
                save_state(state['epoch'], label_columns, train_acc, val_loss, val_acc, target_folder, target_name, model, optimizer, all_targs, all_preds, state['train_losses'], state['valid_losses'],)
                print("Saved checkpoint due to improved val_loss_bag")

                
                state['epoch'] += 1

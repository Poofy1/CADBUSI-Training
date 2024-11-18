import os
import torch.utils.data as TUD
from tqdm import tqdm
from torch import nn
from data.save_arch import *
from util.Gen_ITS2CLR_util import *
import torch.optim as optim
from data.format_data import *
from data.sudo_labels import *
from archs.model_simpleMIL import *
from data.bag_loader import *
from data.instance_loader import *
from loss.FocalLoss import *
from util.eval_util import *
from config import *
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class BCELossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Apply label smoothing
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        # Calculate BCE
        bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))
        return bce.mean()

if __name__ == '__main__':

    # Config
    model_version = '1'
    head_name = "Custom_Instance1"
    data_config = LesionDataConfig  # or LesionDataConfig
    
    config = build_config(model_version, head_name, data_config)
    bags_train, bags_val = prepare_all_data(config)
    num_classes = len(config['label_columns']) + 1
    num_labels = len(config['label_columns'])

    # Create Model
    model = Embeddingmodel(config['arch'], config['pretrained_arch'], num_classes = num_labels).cuda()
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")        
    
    # LOSS INIT
    config.update({
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        'label_smoothing': 0.1,
        'warmup_epochs': 5,
        'cosine_schedule': True
    })

    optimizer = optim.SGD(model.parameters(),
                        lr=config['learning_rate'],
                        momentum=0.9,
                        nesterov=True,
                        weight_decay=0.001) # original .001

    # Mixup function
    def mixup_data(x, y, alpha=1.0):
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    #criterion = BCELossWithSmoothing(smoothing=config['label_smoothing']).cuda()
    BCE_loss = nn.BCELoss()
    
    # MODEL INIT
    model, optimizer, state = setup_model(model, config, optimizer)
    
    # Training loop
    while state['epoch'] < config['total_epochs']:

            
        # Used the instance predictions from bag training to update the Instance Dataloader
        instance_dataset_train = Instance_Dataset(bags_train, state['selection_mask'], transform=train_transform, warmup=True)
        instance_dataset_val = Instance_Dataset(bags_val, state['selection_mask'], transform=val_transform, warmup=True)
        train_sampler = InstanceSampler(instance_dataset_train, config['instance_batch_size'], strategy=1)
        val_sampler = InstanceSampler(instance_dataset_val, config['instance_batch_size'], strategy=1)
        instance_dataloader_train = TUD.DataLoader(instance_dataset_train, batch_sampler=train_sampler, num_workers=4, collate_fn = collate_instance, pin_memory=True)
        instance_dataloader_val = TUD.DataLoader(instance_dataset_val, batch_sampler=val_sampler, collate_fn = collate_instance)
        

        

        print('Training Feature Extractor')
        print(f'Warmup Mode: {state["warmup"]}')
        
        
        for iteration in range(100): 
            losses = AverageMeter()
            instance_total_correct = 0
            total_samples = 0
            model.train()
            train_pred = PredictionTracker()
            
            # In your training loop where you process batches:
            for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_train, total=len(instance_dataloader_train))):
                images = images.cuda(non_blocking=True)
                instance_labels = instance_labels.cuda(non_blocking=True)

                # Apply mixup
                lam = np.random.beta(.2, 1.0)  # alpha is typically 0.2 or 1.0
                batch_size = images.size()[0]
                index = torch.randperm(batch_size).cuda()
                
                # Mix the images
                mixed_images = lam * images + (1 - lam) * images[index, :]
                
                # Mix the labels
                labels_a, labels_b = instance_labels, instance_labels[index]

                # Forward pass
                optimizer.zero_grad()
                _, instance_predictions, _, features = model(mixed_images)
                features = features.to(device)

                # Calculate mixed BCE loss
                loss_a = BCE_loss(instance_predictions, labels_a.float())
                loss_b = BCE_loss(instance_predictions, labels_b.float())
                bce_loss_value = lam * loss_a + (1 - lam) * loss_b

                # Backward pass and optimization
                total_loss = bce_loss_value
                total_loss.backward()
                optimizer.step()

                # Update metrics
                losses.update(total_loss.item(), images.size(0))
                
                # Calculate accuracy considering both original and mixed labels
                predicted = (instance_predictions > 0.5).float()
                correct_a = (predicted == labels_a).float()
                correct_b = (predicted == labels_b).float()
                
                # Weight the correctness by mixup lambda
                correct = lam * correct_a + (1 - lam) * correct_b
                instance_total_correct += correct.sum().item()
                total_samples += instance_labels.numel()
                
                # Store raw predictions and targets
                train_pred.update(instance_predictions, instance_labels, unique_id)

            # Calculate accuracies
            instance_train_acc = instance_total_correct / total_samples
            
            # Validation loop
            model.eval()
            instance_total_correct = 0
            total_samples = 0
            val_losses = AverageMeter()
            val_pred = PredictionTracker()

            with torch.no_grad():
                for idx, (images, instance_labels, unique_id) in enumerate(tqdm(instance_dataloader_val, total=len(instance_dataloader_val))):
                    images = images.cuda(non_blocking=True)
                    instance_labels = instance_labels.cuda(non_blocking=True)

                    # Forward pass
                    _, instance_predictions, _, features = model(images)
                    features.to(device)
                    
                    # Get loss
                    loss_value = BCE_loss(instance_predictions, instance_labels.float())
                    total_loss = loss_value
                    val_losses.update(total_loss.item(), images[0].size(0))

                    # Calculate correct predictions
                    predicted = (instance_predictions > 0.5).float()  # Assuming 0.5 as threshold
                    correct = (predicted == instance_labels).float()
                    instance_total_correct += correct.sum().item()
                    total_samples += instance_labels.numel()
                    
                    # Store raw predictions and targets
                    val_pred.update(instance_predictions, instance_labels, unique_id)

            # Calculate accuracies
            instance_val_acc = instance_total_correct / total_samples
            
            print(f'[{iteration+1}/{100}] Train Loss: {losses.avg:.5f}, Train FC Acc: {instance_train_acc:.5f}')
            print(f'[{iteration+1}/{100}] Val Loss:   {val_losses.avg:.5f}, Val FC Acc: {instance_val_acc:.5f}')
            
            # Save the model
            if val_losses.avg < state['val_loss_instance']:
                state['val_loss_instance'] = val_losses.avg
                state['mode'] = 'instance'
                
                save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, optimizer)
                save_metrics(config, state, train_pred, val_pred)
                print("Saved checkpoint due to improved val_loss_instance")


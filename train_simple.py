import os
from fastai.vision.all import *
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
from torch import nn
from training_eval import *
from torch.optim import Adam
from data_prep import *
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")

class BagOfImagesDataset(Dataset):

    def __init__(self, data, imsize, normalize=True):
        self.bags = data
        self.normalize = normalize
        self.imsize = imsize

        # Normalize
        if normalize:
            self.tsfms = T.Compose([
                T.ToTensor(),
                T.Resize((self.imsize, self.imsize)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tsfms = T.Compose([
                T.ToTensor(),
                T.Resize((self.imsize, self.imsize))
            ])

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        filenames = bag[0]
        labels = bag[1]
        ids = bag[2]
        
        data = torch.stack([
            self.tsfms(Image.open(fn).convert("RGB")) for fn in filenames
        ]).cuda()
        
        # Use the first label from the bag labels (assuming all instances in the bag have the same label)
        label = torch.tensor(labels[0], dtype=torch.float).cuda()

        return data, label


class SimpleMaxPoolAggregator(nn.Module):
    def forward(self, x):
        return x.max(dim=1)[0]  # Max pool along the bag size dimension

class MilResNet(nn.Module):
    def __init__(self, base_model, num_features):
        super(MilResNet, self).__init__()
        self.base_model = base_model
        self.aggregator = SimpleMaxPoolAggregator()
        self.classifier = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, bag_size, num_channels, height, width = x.size()
        x = x.view(-1, num_channels, height, width)
        features = self.base_model(x)
        features = features.view(batch_size, bag_size, -1)
        aggregated_features = self.aggregator(features)
        output = self.classifier(aggregated_features)
        output = self.sigmoid(output)
        return output


if __name__ == '__main__':

    model_name = 'test1'
    img_size = 512
    batch_size = 5
    bag_size = 8
    epochs = 20   
    reg_lambda = 0 #0.001
    lr = 0.001

    print("Preprocessing Data...")
    
    # Load CSV data
    export_location = 'F:/Temp_SSD_Data/export_09_14_2023/'
    case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
    breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
    image_data = pd.read_csv(f'{export_location}/ImageData.csv')
    data = filter_raw_data(breast_data, image_data)

    #Cropping images
    cropped_images = f"{export_location}/temp_cropped/"
    preprocess_and_save_images(data, export_location, cropped_images, img_size)

    # Split the data into training and validation sets
    train_patient_ids = case_study_data[case_study_data['valid'] == 0]['Patient_ID']
    val_patient_ids = case_study_data[case_study_data['valid'] == 1]['Patient_ID']
    train_data = data[data['Patient_ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Patient_ID'].isin(val_patient_ids)].reset_index(drop=True)

    train_bags = create_bags(train_data, bag_size, cropped_images)
    val_bags = create_bags(val_data, bag_size, cropped_images) 
    
    print(f'There are {len(train_data)} files in the training data')
    print(f'There are {len(val_data)} files in the validation data')
    malignant_count, non_malignant_count = count_malignant_bags(train_bags)
    print(f"Number of Malignant Bags: {malignant_count}")
    print(f"Number of Non-Malignant Bags: {non_malignant_count}")
    
    
    
    print("Training Data...")

    # Create datasets
    dataset_train = BagOfImagesDataset(train_bags, img_size)
    dataset_val = BagOfImagesDataset(val_bags, img_size)
        
    # Create data loaders
    train_dl =  DataLoader(dataset_train, batch_size=batch_size, drop_last=True, shuffle = True)
    val_dl =    DataLoader(dataset_val, batch_size=batch_size, drop_last=True)

    
    # RESNET Model
    # Remove the final fully connected layer from the pretrained ResNet-18 model.
    base_model = models.resnet34(pretrained=True)
    num_ftrs = base_model.fc.in_features
    base_model = nn.Sequential(*list(base_model.children())[:-1], nn.Flatten())

    # Create the MIL model.
    model = MilResNet(base_model, num_ftrs).to(device)


    optimizer = Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    
    train_losses_over_epochs = []
    valid_losses_over_epochs = []
    all_targs = []
    all_preds = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for (xb, yb) in tqdm(train_dl, total=len(train_dl)):  
            xb, yb = xb.cuda(), yb.cuda()
            optimizer.zero_grad()
            outputs = model(xb).squeeze()
            loss = loss_func(outputs, yb)
            
            #print(f'loss: {loss}\n pred: {outputs}\n true: {yb}')
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            predicted = torch.round(outputs)
            
            all_targs.extend(yb.cpu().numpy())
            if len(predicted.size()) == 0:
                predicted = predicted.view(1)
            all_preds.extend(predicted.cpu().detach().numpy())

            
            total += yb.size(0)
            correct += predicted.eq(yb.squeeze()).sum().item()

        train_loss = total_loss / total
        train_acc = correct / total


        # Evaluation phase
        model.eval()
        total_val_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for (xb, yb) in tqdm(val_dl, total=len(val_dl)):   
                xb, yb = xb.cuda(), yb.cuda()
                outputs = model(xb).squeeze()
                loss = loss_func(outputs, yb)
                
                total_val_loss += loss.item() * len(xb)
                predicted = torch.round(outputs)
                total += yb.size(0)
                correct += predicted.eq(yb.squeeze()).sum().item()

        val_loss = total_val_loss / total
        val_acc = correct / total
        
        train_losses_over_epochs.append(train_loss)
        valid_losses_over_epochs.append(val_loss)
        
        print(f"Epoch {epoch+1} | Acc   | Loss")
        print(f"Train   | {train_acc:.4f} | {train_loss:.4f}")
        print(f"Val     | {val_acc:.4f} | {val_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), f"{env}/models/{model_name}.pth")

    # Save the loss graph
    plot_loss(train_losses_over_epochs, valid_losses_over_epochs, f"{env}/models/{model_name}_loss.png")
    
    # Save the confusion matrix
    vocab = ['not malignant', 'malignant']  # Replace with your actual vocab
    plot_Confusion(all_targs, all_preds, vocab, f"{env}/models/{model_name}_confusion.png")

        
    
import pandas as pd
import torch, os
from torch.utils.data import Dataset
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from PIL import Image
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")

# Load CSV data
export_location = f'F:/Temp_SSD_Data/export_08_18_2023/'
case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
image_data = pd.read_csv(f'{export_location}/ImageData.csv')

# Join dataframes on PatientID
data = pd.merge(breast_data, image_data, left_on=['Patient_ID', 'Breast'], right_on=['Patient_ID', 'laterality'], suffixes=('', '_image_data'))

# Remove columns from image_data that also exist in breast_data
for col in breast_data.columns:
    if col + '_image_data' in data.columns:
        data.drop(col + '_image_data', axis=1, inplace=True)
        

class ResizeAndPad:
    def __init__(self, output_size, fill=0):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if h > w:
            new_h, new_w = self.output_size, int(self.output_size * (w / h))
        else:
            new_h, new_w = int(self.output_size * (h / w)), self.output_size
        img = transforms.functional.resize(img, (new_h, new_w))

        diff = self.output_size - new_w if h > w else self.output_size - new_h
        padding = [diff // 2, diff // 2]

        # If the difference is odd, add the extra padding to the end
        if diff % 2 != 0:
            padding[1] += 1

        # Use the padding values for the left/right or top/bottom
        padding = (padding[0], 0, padding[1], 0) if h > w else (0, padding[0], 0, padding[1])
        img = transforms.functional.pad(img, padding, fill=self.fill)
        return img

def Get_Image_Labels(same_patient_data, i):
    ori_case = {
        'long': 0,
        'trans': 1,
    }
    orientation = same_patient_data.iloc[i]['orientation']
    orientation = ori_case.get(orientation, -1)
    
    return torch.tensor([orientation])

class CASBUSI_Dataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.patient_ids = {idx: pid for idx, pid in enumerate(data['Patient_ID'].unique())}

    def __len__(self):
        return len(self.patient_ids)  # Return number of unique patients (bags)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        same_patient_data = self.data[self.data['Patient_ID'] == patient_id].reset_index(drop=True)
        bag_size = len(same_patient_data)
        
        if bag_size > 50:
            return None, None
        
        images = []
        extra_data_list = []
        for i in range(bag_size):
            img_name = os.path.join(self.root_dir, same_patient_data.iloc[i]['ImageName'])
            extra_data = Get_Image_Labels(same_patient_data, i)
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            images.append(image)
            extra_data_list.append(extra_data)

        images = torch.stack(images)
        extra_data_list = torch.stack(extra_data_list)

        has_unknown = same_patient_data.loc[0, 'Has_Unknown']
        if not has_unknown:
            target = torch.tensor(same_patient_data.loc[0, ['Has_Malignant', 'Has_Benign']].values.astype('float'))
        else:
            target = None

        return (images, extra_data_list), target


def load_model(model, model_path):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print("No previous model found, starting training from scratch.")
    return model

def collate_fn(batch):
    return batch

def train_model(model, model_name, epochs):
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    print_every = 100
    
    for epoch in range(epochs):  # Number of epochs
        model.train()
        running_loss = 0.0
        running_corrects = 0
        running_total = 0
        
        final_loss = 0.0
        final_corrects = 0
        final_total = 0
        for i, data in enumerate(train_loader):
            torch.cuda.empty_cache()
            bag, labels = data[0]  # Get the first item in the batch
            if labels is None:  # Skip bags with label None
                continue
            labels = labels.squeeze().to(device)   # Remove extra dimension from labels

            bag_outputs = []
            images, extra_data_list = bag  # Unpack bag into images and extra data
            for image, extra_data in zip(images, extra_data_list):
                image = image.unsqueeze(0).to(device)   # Add batch dimension
                extra_data = extra_data.to(device)
                output = model(image, extra_data)
                bag_outputs.append(output)
                
            # Aggregate bag outputs
            outputs = torch.stack(bag_outputs).mean(dim=0).squeeze(0)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            final_loss += loss.item()

            # Calculate accuracy
            predicted = torch.round(torch.sigmoid(outputs))  # Convert to 0/1
            running_total += labels.size(0)
            running_corrects += (predicted == labels).sum().item()
            final_total += labels.size(0)
            final_corrects += (predicted == labels).sum().item()

            # Print every 'print_every' mini-batches
            if i % print_every == (print_every-1):  # print every 'print_every' mini-batches
                avg_loss = running_loss / print_every
                avg_acc = running_corrects / running_total  # Multiply by label size to account for multiple labels per image
                print(f'Epoch {epoch+1}, Step {i+1}/{len(train_loader)} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
                running_loss = 0.0  # reset running loss
                running_corrects = 0  # reset running corrects
                running_total = 0  # reset total

        train_acc = final_corrects / final_total

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                torch.cuda.empty_cache()
                bag, labels = data[0]  # Get the first item in the batch
                if labels is None:  # Skip bags with label None
                    continue
                labels = labels.squeeze().to(device)   # Remove extra dimension from labels

                bag_outputs = []
                for image in bag:
                    image = image.unsqueeze(0).to(device)   # Add batch dimension
                    output = model(image)
                    bag_outputs.append(output)
                    torch.cuda.empty_cache()

                # Aggregate bag outputs
                outputs = torch.stack(bag_outputs).mean(dim=0).squeeze(0)

                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                # Calculate accuracy
                predicted = torch.round(torch.sigmoid(outputs))  # Convert to 0/1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        train_loss = final_loss/len(train_loader)
        val_loss = val_running_loss/len(val_loader)
        
        print(f"Epoch {epoch+1} | Acc | Loss")
        print(f"Train | {train_acc:.4f} | {train_loss:.4f}")
        print(f"Val | {val_acc:.4f} | {val_loss:.4f}")
                
    torch.save(model.state_dict(), f'{env}/models/{model_name}.pth')



class Model(nn.Module):
    def __init__(self, num_classes=2, linear_input_size=64):
        super(Model, self).__init__()
        
        # Define a convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Define an adaptive pooling layer to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))
        
        # Define a fully connected layer for classification
        self.fc_image = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*10*10, 128),
            nn.ReLU(inplace=True)
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(128 + linear_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, linear_input):
        image_features = self.features(image)
        image_features = self.adaptive_pool(image_features)
        image_features = self.fc_image(image_features)
        
        # Add an extra dimension to linear_input
        linear_input = linear_input.unsqueeze(1)
        
        # Concatenate the image features with the linear input
        combined_features = torch.cat((image_features, linear_input), dim=1)
        
        output = self.fc_combined(combined_features)
        return output
    
    
if __name__ == "__main__":
    
    model_name = 'model_09_02_2023'
    epochs = 3
    image_size = 512
    
    # RESNET Model
    """model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.conv1 = torch.nn.Conv2d(1, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(num_ftrs, 2)"""

    model = Model(num_classes=2, linear_input_size=1)
    model = model.to(device)
    
    # Define transformations
    train_transform = transforms.Compose([
        ResizeAndPad(image_size),  # Resize
        transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])

    val_transform = transforms.Compose([
        ResizeAndPad(image_size),  # Resize
        transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])

    # Split the data into training and validation sets
    train_patient_ids = case_study_data[case_study_data['valid'] == 0]['Patient_ID']
    val_patient_ids = case_study_data[case_study_data['valid'] == 1]['Patient_ID']
    train_data = data[data['Patient_ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Patient_ID'].isin(val_patient_ids)].reset_index(drop=True)

    # Create datasets
    train_dataset = CASBUSI_Dataset(train_data, f'{export_location}/images/', transform=train_transform)
    val_dataset = CASBUSI_Dataset(val_data, f'{export_location}/images/', transform=val_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers = 8, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers = 4, persistent_workers=True)
    
    os.makedirs(f"{env}/models/", exist_ok=True)
    model = load_model(model, f"{env}/models/{model_name}.pt")
    #summary(model, input_size=(1, 1, image_size, image_size))
    print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

    train_model(model, model_name, epochs)
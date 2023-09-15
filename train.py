import pandas as pd
import torch, os
from torch.utils.data import Dataset
from tqdm import tqdm
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
env = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda")

# Load CSV data
export_location = 'F:/Temp_SSD_Data/export_09_14_2023/'
case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
image_data = pd.read_csv(f'{export_location}/ImageData.csv')

# Join dataframes on PatientID
data = pd.merge(breast_data, image_data, left_on=['Patient_ID', 'Breast'], right_on=['Patient_ID', 'laterality'], suffixes=('', '_image_data'))

# Remove columns from image_data that also exist in breast_data
for col in breast_data.columns:
    if col + '_image_data' in data.columns:
        data.drop(col + '_image_data', axis=1, inplace=True)


def process_single_image(row, root_dir, output_dir, resize_and_pad):
    patient_id = row['Patient_ID']
    img_name = row['ImageName']
    input_path = os.path.join(f'{root_dir}images/', img_name)
    output_path = os.path.join(output_dir, img_name)

    if os.path.exists(output_path):  # Skip images that are already processed
        return

    image = Image.open(input_path)
    image = resize_and_pad(image)
    image.save(output_path)

def preprocess_and_save_images(data, root_dir, output_dir, image_size, fill=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Preprocessing Data")

    resize_and_pad = ResizeAndPad(image_size, fill)
    data_rows = [row for _, row in data.iterrows()]  # Convert the DataFrame to a list of rows

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_image, row, root_dir, output_dir, resize_and_pad): row for row in data_rows}

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                future.result()  # We don't actually use the result, but this will raise any exceptions
                pbar.update()

class GrayscaleToRGB:
    def __call__(self, img):
        if len(img.getbands()) == 1:  # If image is grayscale
            img = transforms.functional.to_pil_image(np.stack([img] * 3, axis=-1))
        return img

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
        for i in range(bag_size):
            img_name = os.path.join(self.root_dir, same_patient_data.iloc[i]['ImageName'])
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images)

        has_unknown = same_patient_data.loc[0, 'Has_Unknown']
        if not has_unknown:
            target = torch.tensor(same_patient_data.loc[0, ['Has_Malignant', 'Has_Benign']].values.astype('float'))
        else:
            target = None

        return images, target


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
            bag, labels = data[0]  # Get the first item in the batch
            if labels is None:  # Skip bags with label None
                continue
            print(data)
            labels = labels.squeeze().to(device, non_blocking=True)

            bag = bag.to(device, non_blocking=True)
            
            # Get the output for the entire bag in one forward pass
            outputs = model(bag).mean(dim=0).squeeze(0)

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
        with torch.no_grad():
            val_running_loss = 0.0
            correct = 0
            total = 0

            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                bag, labels = data[0]  # Get the first item in the batch
                if labels is None:  # Skip bags with label None
                    continue

                labels = labels.squeeze().to(device, non_blocking=True)
                bag = bag.to(device, non_blocking=True)

                # Get the output for the entire bag in one forward pass
                outputs = model(bag).mean(dim=0).squeeze(0)

                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                # Calculate accuracy
                predicted = torch.round(torch.sigmoid(outputs))  # Convert to 0/1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_acc = correct / total if total > 0 else 0.0  # Prevent division by zero
            val_loss = val_running_loss / total if total > 0 else 0.0  # Prevent division by zero

            print(f"Epoch {epoch+1} | Acc | Loss")
            print(f"Train | {train_acc:.4f} | {final_loss / len(train_loader):.4f}")
            print(f"Val | {val_acc:.4f} | {val_loss:.4f}")
            
    torch.save(model.state_dict(), f'{env}/models/{model_name}.pth')



class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        
        # Define a convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
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
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, image):
        image_features = self.features(image)
        image_features = self.adaptive_pool(image_features)
        image_features = self.fc_image(image_features)
        
        output = self.fc_combined(image_features)
        return output
    
    
if __name__ == "__main__":
    
    model_name = 'model_09_02_2023'
    epochs = 3
    image_size = 512
    
    
    
    # RESNET Model
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.conv1 = torch.nn.Conv2d(3, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(num_ftrs, 2)

    #model = Model(num_classes=2)
    model = model.to(device)
    
    #Preparing data
    cropped_images = f"{export_location}/temp_cropped/"
    preprocess_and_save_images(data, export_location, cropped_images, image_size)
        
    
    
    # Define transformations
    train_transform = transforms.Compose([
        GrayscaleToRGB(),
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])

    val_transform = transforms.Compose([
        GrayscaleToRGB(), 
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])

    # Split the data into training and validation sets
    train_patient_ids = case_study_data[case_study_data['valid'] == 0]['Patient_ID']
    val_patient_ids = case_study_data[case_study_data['valid'] == 1]['Patient_ID']
    train_data = data[data['Patient_ID'].isin(train_patient_ids)].reset_index(drop=True)
    val_data = data[data['Patient_ID'].isin(val_patient_ids)].reset_index(drop=True)

    # Create datasets
    train_dataset = CASBUSI_Dataset(train_data, f'{cropped_images}/', transform=train_transform)
    val_dataset = CASBUSI_Dataset(val_data, f'{cropped_images}/', transform=val_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=8, persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4, persistent_workers=True, pin_memory=True)

    
    os.makedirs(f"{env}/models/", exist_ok=True)
    model = load_model(model, f"{env}/models/{model_name}.pt")
    #summary(model, input_size=(1, 1, image_size, image_size))
    print(f'Total model parameters: {sum(p.numel() for p in model.parameters())}')

    train_model(model, model_name, epochs)
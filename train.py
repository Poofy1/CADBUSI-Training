import pandas as pd
import torch, os
from torch.utils.data import Dataset
from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt
from torchvision import transforms
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
        
        
data.to_csv(f'{env}/test.csv')

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

# Define transformations
train_transform = transforms.Compose([
    ResizeAndPad(256),  # Resize the shortest side to 256
    transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])

val_transform = transforms.Compose([
    ResizeAndPad(256),  # Resize the shortest side to 256
    transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])

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
        
        bag = []
        for i in range(bag_size):
            img_name = os.path.join(self.root_dir, same_patient_data.iloc[i]['ImageName'])
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            
            # preview what the images look like for the model
            """img_disp = image.clone().detach()  
            img_disp = img_disp * 0.5 + 0.5
            if img_disp.shape[0] == 1:
                img_disp = img_disp.squeeze(0)
            plt.imshow(img_disp.numpy(), cmap='gray')
            plt.show()"""
            
            bag.append(image)

        bag = torch.stack(bag)
        labels = torch.tensor(same_patient_data.loc[0, ['Has_Malignant', 'Has_Benign', 'Has_Unknown']].values.astype('float'))

        return bag, labels
    
def load_model(model, model_path):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print("No previous model found, starting training from scratch.")
    return model

def collate_fn(batch):
    return batch

def pad_image(img):
    return transforms.functional.pad(img, (0, 0, max(0, 256 - img.size[0]), max(0, 256 - img.size[1])), fill=0)
  

# Define model
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2, stride=2),
    torch.nn.Flatten(),
    torch.nn.Linear(64*64*64, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 3),
)

def train_model(model, model_name, epochs):
    
    # Define loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):  # Number of epochs
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            bag, labels = data[0]  # Get the first item in the batch
            labels = labels.squeeze().to(device)   # Remove extra dimension from labels

            bag_outputs = []
            for image in bag:
                image = image.unsqueeze(0).to(device)   # Add batch dimension
                output = model(image)
                bag_outputs.append(output)

            # Aggregate bag outputs
            outputs = torch.stack(bag_outputs).mean(dim=0).squeeze(0)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predicted = torch.round(torch.sigmoid(outputs))  # Convert to 0/1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                bag, labels = data[0]  # Get the first item in the batch
                labels = labels.squeeze().to(device)   # Remove extra dimension from labels

                bag_outputs = []
                for image in bag:
                    image = image.unsqueeze(0).to(device)   # Add batch dimension
                    output = model(image)
                    bag_outputs.append(output)

                # Aggregate bag outputs
                outputs = torch.stack(bag_outputs).mean(dim=0).squeeze(0)

                loss = criterion(outputs, labels)
                val_running_loss += loss.item()

                # Calculate accuracy
                predicted = torch.round(torch.sigmoid(outputs))  # Convert to 0/1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        train_loss = running_loss/len(train_loader)
        val_loss = val_running_loss/len(val_loader)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
    torch.save(model.state_dict(), f'{env}/{model_name}.pth')



if __name__ == "__main__":
    
    model_name = 'model_09_02_2023'
    epochs = 10
    image_size = 256
    model = model.to(device)

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
    
    os.makedirs(f"{env}/model/", exist_ok=True)
    model = load_model(model, f"{env}/model/{model_name}.pt")
    summary(model, input_size=(1, 1, image_size, image_size))

    train_model(model, model_name, epochs)
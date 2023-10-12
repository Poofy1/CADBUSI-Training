import os
from fastai.vision.all import *
from torchvision import datasets
from model_ABMIL import *
from train_loop import *
from data_prep import *
env = os.path.dirname(os.path.abspath(__file__))



def load_trained_model(model_path, encoder_arch):
    encoder = create_timm_body(encoder_arch)
    nf = num_features_model(nn.Sequential(*encoder.children()))
    aggregator = ABMIL_aggregate(nf=nf, num_classes=1, pool_patches=3, L=128)
    bagmodel = EmbeddingBagModel(encoder, aggregator).cuda()
    bagmodel.load_state_dict(torch.load(model_path))
    bagmodel.eval()
    return bagmodel




def predict_on_test_set(model, test_dl):
    bag_predictions = []
    
    with torch.no_grad():
        for (data, yb) in tqdm(test_dl, total=len(test_dl)): 
            xb, ids = data  
            xb, ids, yb = xb.cuda(), ids.cuda(), yb.cuda()
            
            outputs = model((xb, ids)).squeeze(dim=1)
            
            # For the bag-level prediction, we can take the maximum prediction across images in the bag.
            bag_pred = torch.max(outputs).item()
            bag_predictions.append(bag_pred)
    
    return bag_predictions


def generate_test_filenames_from_folders(test_root_path):
    bag_files = []
    bag_ids = []
    current_id = 0
    
    for subdir, _, files in os.walk(test_root_path):
        if files:
            current_bag = [os.path.join(subdir, f) for f in files if f.endswith(('.jpg', '.png'))]
            bag_files.extend(current_bag)
            
            # Assign the same ID for all images in this bag
            bag_ids.extend([current_id] * len(current_bag))
            
            current_id += 1  # Increment the ID for the next bag
    
    return bag_files, bag_ids

# Config
model_name = 'MixupTest2'
encoder_arch = 'resnet50'
img_size = 256
min_bag_size = 2
max_bag_size = 15

# Paths
export_location = 'D:/DATA/CASBUSI/exports/export_09_28_2023/'
case_study_data = pd.read_csv(f'{export_location}/CaseStudyData.csv')
breast_data = pd.read_csv(f'{export_location}/BreastData.csv')
image_data = pd.read_csv(f'{export_location}/ImageData.csv')
cropped_images = f"F:/Temp_SSD_Data/{img_size}_images/"


# Load the trained model
model_path = f'{env}/models/{model_name}/{model_name}.pth'
model = load_trained_model(model_path, encoder_arch)



# Load test data
_, _, _, files_val, ids_val, labels_val = prepare_all_data(export_location, case_study_data, breast_data, image_data, 
                                                                                        cropped_images, img_size, min_bag_size, max_bag_size)


# 1. Extract unique IDs
unique_ids = np.unique(ids_val)

# 2. Randomly select 5 of them
selected_ids = np.random.choice(unique_ids, 5, replace=False)

selected_files = []
selected_ids_list = []
selected_labels = []

# 3. For each selected ID, extract the corresponding file paths and labels
for idx, uid in enumerate(selected_ids):
    mask = (np.array(ids_val) == uid)
    
    # For labels, directly append the label corresponding to the unique ID
    selected_labels.append(labels_val[uid])
    
    # For files and IDs
    selected_files.extend(list(np.array(files_val)[mask]))
    
    # Use idx instead of the actual id to ensure ids start from 0 and are contiguous
    selected_ids_list.extend([idx] * sum(mask))

selected_files = np.array(selected_files)
selected_ids_list = np.array(selected_ids_list)
selected_labels = np.array(selected_labels)

# Now, create your dataset and dataloader using selected_files, selected_ids_list, and selected_labels
dataset_val = BagOfImagesDataset(selected_files, selected_ids_list, selected_labels, train=False)
val_dl = TUD.DataLoader(dataset_val, batch_size=1, collate_fn=collate_custom, drop_last=True)

# Make predictions on test set
predictions = predict_on_test_set(model, val_dl)

print(predictions)

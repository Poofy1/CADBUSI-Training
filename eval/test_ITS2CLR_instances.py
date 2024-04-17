import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastai.vision.all import *
from archs.model_ABMIL import *
from train_GenSCL_ITS2CLR import *
from data.format_data import *
import matplotlib.pyplot as plt

val_transform = T.Compose([
            CLAHETransform(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class Instance_Dataset(TUD.Dataset):
    def __init__(self, bags_dict, selection_mask, transform=None, warmup=True):
        self.transform = transform
        self.warmup = warmup 
        self.images = []
        self.final_labels = []
        self.warmup_mask = []
        self.bag_ids = []
        
        

        for bag_id, bag_info in bags_dict.items():
            images = bag_info['images']
            image_labels = bag_info['image_labels']
            bag_label = bag_info['bag_labels'][0]  # Assuming each bag has a single label
            
            bag_id_key = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
            
            
            if bag_id_key in selection_mask:
                selection_mask_labels, _ = selection_mask[bag_id_key]
            else: 
                selection_mask_labels = None
                
            #print(selection_mask_labels)
            
            for idx, (img, label) in enumerate(zip(images, image_labels)):
                image_label = None
                warmup_mask_value = 0
                
                if not self.warmup:
                    # Only include confident instances (selection_mask) or negative bags or instance labels
                    if label[0] is not None:
                        image_label = label[0]
                    elif selection_mask_labels is not None and selection_mask_labels[idx] != -1:
                        image_label = selection_mask_labels[idx]
                    elif bag_label == 0:
                        image_label = 0
                else:
                    # Include all data but replace with image_labels if present
                    if label[0] is not None:
                        image_label = label[0]
                    else:
                        image_label = bag_label  # Use bag label if instance label is not present
                        if bag_label == 1:
                            warmup_mask_value = 1 # Set warmup mask to 1 for instances without label[0] and bag_label is 1
                
                if image_label is not None:
                    self.images.append(img)
                    self.final_labels.append(image_label)
                    self.warmup_mask.append(warmup_mask_value)
                    self.bag_ids.append(bag_id_key)

    def __getitem__(self, index):
        img_path = self.images[index]
        instance_label = self.final_labels[index]
        warmup_unconfident = self.warmup_mask[index]
        ids = self.bag_ids[index]
        
        img = Image.open(img_path).convert("RGB")
        image_data_q = self.transform(img)

        return image_data_q, instance_label, warmup_unconfident, ids


    def __len__(self):
        return len(self.images)
    
def collate_instance(batch):
    batch_data_q = []
    batch_labels = []
    batch_unconfident = []
    batch_ids = []

    for image_data_q, bag_label, warmup_unconfident, ids in batch:
        batch_data_q.append(image_data_q)
        batch_labels.append(bag_label)
        batch_unconfident.append(warmup_unconfident)
        batch_ids.append(ids)

    # Stack the images and labels
    batch_data_q = torch.stack(batch_data_q)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)
    batch_unconfident = torch.tensor(batch_unconfident, dtype=torch.long)
    batch_ids = torch.tensor(batch_ids, dtype=torch.long)

    return batch_data_q, batch_labels, batch_unconfident, batch_ids

def test_dataset(output_path, label_columns, instance_columns):
    # Load data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)

    # Combine training and validation data
    combined_dict = bags_train
    combined_dict.update(bags_val)

    # Now use the combined data for the dataset
    #dataset_combined = TUD.Subset(BagOfImagesDataset(combined_dict, transform=val_transform, save_processed=False),list(range(0,50)))
    #dataset_combined = BagOfImagesDataset(combined_dict, transform=val_transform, save_processed=False)
    #dataset_combined = TUD.Subset(Instance_Dataset(combined_dict, [], transform=val_transform, warmup=True),list(range(0,10000)))
    dataset_combined = Instance_Dataset(combined_dict, [], transform=val_transform, warmup=True)
    combined_dl = TUD.DataLoader(dataset_combined, batch_size=1, collate_fn = collate_instance, drop_last=True)
    
    bag_data = {}
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images, instance_labels, unconfident_mask, id) in tqdm(combined_dl, total=len(combined_dl)):
            xb, yb = [images], instance_labels.cuda()
            
            outputs, instance_pred, features = model(xb, pred_on=True)
            
            outputs = outputs.squeeze(0)
            yb = yb.to(outputs.dtype)
            
            loss = criterion(outputs, yb)
            bag_id = id.item()
            
            if bag_id not in bag_data:
                bag_data[bag_id] = {
                    'instance_predictions': [],
                    'features': np.empty((0, features.shape[1])),
                    'losses': [],
                    'labels': [],
                    'unconfident_mask': []
                }

            bag_data[bag_id]['losses'].append(loss.item())
            bag_data[bag_id]['labels'].append(yb.cpu().numpy().tolist())
            bag_data[bag_id]['features'] = np.concatenate((bag_data[bag_id]['features'], features.cpu().numpy()))
            bag_data[bag_id]['instance_predictions'].append(outputs)
            bag_data[bag_id]['unconfident_mask'].append(unconfident_mask)
    
    # Save the bag_data dictionary to disk
    with open(f'{output_path}/bag_data.pkl', 'wb') as f:
        pickle.dump(bag_data, f)
    
    return bag_data
    

if __name__ == '__main__':

    # Config
    model_name = 'cifar10_Res18_03'
    dataset_name = 'cifar10'
    label_columns = ['Has_Truck']
    instance_columns = ['']   #['Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 32
    min_bag_size = 2
    max_bag_size = 25
    use_efficient_net = False
    model_folder = f"{env}/models/{model_name}/"
    lr = 0.001


    # Paths
    export_location = f'D:/DATA/CASBUSI/exports/{dataset_name}/'
    cropped_images = f"F:/Temp_SSD_Data/{dataset_name}_{img_size}_images/"
    output_path = f"{parent_dir}/results/{model_name}_DatasetTest/"
    mkdir(output_path, exist_ok=True)

    # Get Training Data
    num_labels = len(label_columns)

    


    if os.path.exists(f'{output_path}/bag_data.pkl'):
        with open(f'{output_path}/bag_data.pkl', 'rb') as f:
            bag_data = pickle.load(f)
    else:
        
        # Get Model
        if use_efficient_net:
            encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            nf = 512
            # Replace the last fully connected layer with a new one
            num_features = encoder.classifier[1].in_features
            encoder.classifier[1] = nn.Linear(num_features, nf)
            
        else:
            encoder = create_timm_body("resnet18")
            nf = num_features_model( nn.Sequential(*encoder.children()))
        

        model = Embeddingmodel(encoder = encoder, nf = nf, num_classes = num_labels, efficient_net = use_efficient_net).cuda()
        
        
        
        
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")        
            
        BCE_loss = nn.BCELoss()
        genscl = GenSupConLossv2(temperature=0.07, contrast_mode='all', base_temperature=0.07)
        
        # Check if the model already exists
        model_folder = f"{parent_dir}/models/{model_name}/"
        model_path = f'{model_folder}/{model_name}.pth'
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded pre-existing model from {model_name}")


        bag_data = test_dataset(output_path, label_columns, instance_columns)
        
    
    
    # Find the worst performing images with unconfidence 0
    worst_performing_images = []
    num_worst_images = 10  # Adjust this value to the desired number of worst performing images

    for bag_id, data in bag_data.items():
        instance_predictions = data['instance_predictions']
        labels = data['labels']
        unconfident_mask = data['unconfident_mask']
        
        for i, (pred, label, unconfident) in enumerate(zip(instance_predictions, labels, unconfident_mask)):
            if unconfident.item() == 0:  # Check if unconfidence is 0
                pred_value = pred.item()  # Convert tensor to scalar value
                label_value = label[0]  # Assuming labels are in the format [label_value]
                
                # Calculate the absolute difference between prediction and label
                diff = abs(pred_value - label_value)
                
                worst_performing_images.append((diff, bag_id, i))

    # Sort the worst performing images based on the absolute difference in descending order
    worst_performing_images.sort(reverse=True)

    # Print the worst performing images with unconfidence 0
    print("Worst Performing Images with Unconfidence 0:")
    for diff, bag_id, img_increment in worst_performing_images[:num_worst_images]:
        print(f"Bag ID: {bag_id}, Image Increment: {img_increment}, Difference: {diff:.4f}")
        
    
    # GETTING UMAP
    import umap
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    # Create a new array containing all features, instance predictions, and corresponding labels
    all_features = []
    all_instance_predictions = []
    all_labels = []
    all_unconfident = []

    for bag_id, data in bag_data.items():
        all_features.append(data['features'])
        all_instance_predictions.extend([pred.cpu() for pred in data['instance_predictions']])  # Move tensors to CPU
        all_labels.extend([data['labels'][0][0]] * len(data['features']))
        all_unconfident.extend(data['unconfident_mask'])

    all_features = np.concatenate(all_features)
    all_instance_predictions = np.concatenate([pred.numpy() for pred in all_instance_predictions])  # Convert tensors to NumPy arrays
    all_labels = np.ravel(all_labels)
    all_unconfident = np.ravel(all_unconfident)

    print(all_features.shape)

    # Apply UMAP to the features
    umap_model = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
    features_2d = umap_model.fit_transform(all_features)

    print(features_2d.shape)

    # Create combined labels array
    combined_labels = np.zeros_like(all_labels, dtype=int)
    combined_labels[(all_unconfident == 0) & (all_labels == 0)] = 0  # Confident Negative
    combined_labels[(all_unconfident == 1) & (all_instance_predictions < 0.5)] = 1  # Unconfident Negative
    combined_labels[(all_unconfident == 0) & (all_labels == 1)] = 2  # Confident Positive
    combined_labels[(all_unconfident == 1) & (all_instance_predictions >= 0.5)] = 3  # Unconfident Positive

    # Calculate the silhouette score
    silhouette_avg = silhouette_score(features_2d, combined_labels)
    print(f"Silhouette score: {silhouette_avg}")

    # Downsample the data points
    num_samples = 5000  # Adjust this value based on your desired number of points to plot
    random_indices = np.random.choice(features_2d.shape[0], num_samples, replace=False)
    features_subset = features_2d[random_indices]
    combined_labels_subset = combined_labels[random_indices]

    # Create a 2D scatter plot using Matplotlib
    colors = ['blue', 'lightblue', 'red', 'lightcoral']
    labels = ['Confident Negative', 'Unconfident Negative', 'Confident Positive', 'Unconfident Positive']

    plt.figure(figsize=(8, 6))
    for label, color in zip(range(4), colors):
        plt.scatter(features_subset[combined_labels_subset == label, 0],
                    features_subset[combined_labels_subset == label, 1],
                    c=color, label=labels[label], alpha=0.8, s=10)

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('2D Scatter Plot of Features')
    plt.legend()
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(f'{output_path}/features_2d_plot_labeled_unconfident_umap.png', dpi=300)
    plt.show()
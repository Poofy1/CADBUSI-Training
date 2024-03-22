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


def test_dataset(output_path, label_columns, instance_columns):
    # Load data
    bags_train, bags_val = prepare_all_data(export_location, label_columns, instance_columns, cropped_images, img_size, min_bag_size, max_bag_size)

    # Combine training and validation data
    combined_dict = bags_train
    combined_dict.update(bags_val)

    # Now use the combined data for the dataset
    #dataset_combined = TUD.Subset(BagOfImagesDataset(combined_dict, transform=val_transform, save_processed=False),list(range(0,50)))
    dataset_combined = BagOfImagesDataset(combined_dict, transform=val_transform, save_processed=False)
    combined_dl = TUD.DataLoader(dataset_combined, batch_size=1, collate_fn = collate_bag, drop_last=True)
    
    bag_data = {}
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (data, yb, instance_yb, id) in tqdm(combined_dl, total=len(combined_dl)):
            xb, yb = data, yb.cuda()
            
            outputs, instance_pred, features = model(xb, pred_on=True)
            loss = criterion(outputs, yb)
            bag_id = id.item()
            
            if bag_id not in bag_data:
                bag_data[bag_id] = {
                    'predictions': [],
                    'instance_predictions': [],
                    'features': np.empty((0, features.shape[1])),
                    'losses': [],
                    'labels': []
                }

                bag_data[bag_id]['predictions'].append(outputs.cpu().tolist())
                bag_data[bag_id]['losses'].append(loss.item())
                bag_data[bag_id]['labels'].append(yb.cpu().numpy().tolist())
                bag_data[bag_id]['features'] = np.concatenate((bag_data[bag_id]['features'], features.cpu().numpy()))
                bag_data[bag_id]['instance_predictions'].append(instance_pred)
    
    # Save the bag_data dictionary to disk
    with open(f'{output_path}/bag_data.pkl', 'wb') as f:
        pickle.dump(bag_data, f)
    
    return bag_data
    

if __name__ == '__main__':

    # Config
    model_name = '03_18_2024_Res50_01'
    dataset_name = 'export_03_18_2024'
    label_columns = ['Has_Malignant']
    instance_columns = ['Malignant Lesion Present']   #['Only Normal Tissue', 'Cyst Lesion Present', 'Benign Lesion Present', 'Malignant Lesion Present']
    img_size = 300
    min_bag_size = 2
    max_bag_size = 25
    instance_batch_size =  30
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

    # Get Model
    if use_efficient_net:
        encoder = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        nf = 512
        # Replace the last fully connected layer with a new one
        num_features = encoder.classifier[1].in_features
        encoder.classifier[1] = nn.Linear(num_features, nf)
        
    else:
        encoder = create_timm_body("resnet50")
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



    if os.path.exists(f'{output_path}/bag_data.pkl'):
        with open(f'{output_path}/bag_data.pkl', 'rb') as f:
            bag_data = pickle.load(f)
    else:
        bag_data = test_dataset(output_path, label_columns, instance_columns)
        
    
    
    # Analyzing Data
    
    
    # Create a new array containing all features and corresponding labels
    all_features = []
    all_labels = []
    for bag_id, data in bag_data.items():
        all_features.append(data['features'])
        all_labels.extend([data['labels'][0][0]] * len(data['features']))

    all_features = np.concatenate(all_features)
    all_labels = np.ravel(all_labels)
    
    print(all_features.shape)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=100)
    features_pca = pca.fit_transform(all_features)
    
    print(features_pca.shape)

    # Randomly sample a subset of data points
    num_samples = 5000  # Adjust this based on your needs
    random_indices = np.random.choice(features_pca.shape[0], num_samples, replace=False)
    features_subset = features_pca[random_indices]
    labels_subset = all_labels[random_indices]

    print(features_subset.shape)
    
    # Apply t-SNE to the subset of features
    tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, n_iter=1000)
    features_3d = tsne.fit_transform(features_subset)
    
    print(features_3d.shape)

    # Create a 3D scatter plot of the features, color-coded by labels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['red', 'blue']
    labels = ['Negative', 'Positive']

    for label in [0, 1]:
        mask = labels_subset == label
        ax.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2], c=colors[label], label=labels[label])

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/features_3d_plot_labeled.png')
    plt.show()
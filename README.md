# CADBUSI-Training

ML framework that supports various deep learning architectures for ultrasound image analysis. Research is still underway 

## Requirements

- Python 3.8
- Nvidia GPU
- CUDA 11.3 or greater
- Custom storage package: [storage-adapter](https://github.com/Poofy1/storage-adapter.git) (automatically installed via requirements.txt)

## Usage 
- Clone repository: `git clone https://github.com/Poofy1/CADBUSI-Training.git`
- Install required Python packages with pip: `pip install -r requirements.txt` or on GCP: `pip install --user --only-binary :all: -r requirements.txt`
- Configure `./config.py`
- Generate the [Test Dataset](#test-dataset) or prepare [Custom Data](#dataloader-input-format)
- You can begin training by running one of the provided training scripts in the immediate directory. Each script has its own training loop for that specific technique
     - If you are running a training script on newly processed data, the script will crop and save the images into a temporary `cropped_images` directory
     - This will automatically continue training if a model exists under the specified name already
     - This will automatically unbalanced data by downsampling the majority class until they are balanced (Only when training on 2 classes)
     - Trained models will be saved in the `./models/` dir with your 




## Model Format

### Warmup Phase
- Instance-level training only
- Checkpoints saved in immediate model directory

### Post-Warmup Training
- Alternates between instance and MIL-level learning ([ITS2CLR](https://arxiv.org/pdf/2210.09452.pdf))
- Finetunes the warmup checkpoint
- Often involves generating / using sudo labels

### Versioning Structure
```
./
└── models/
    └── test_run_1/            # Model name
         ├── 1/                # First training version
         ├── 2/                # Second training version
         ├── model.pth         # Warmup checkpoint
         └── ...               # Additional state information
```

This structure allows multiple training iterations to branch from the same warmup checkpoint, enabling experimentation with different strategies while maintaining a common foundation. Within this direcotry are other metric data like accuracy logs, loss graphs, roc graphs, and confuision matrices.



## Model Testing

Testing scripts reside in the `./eval/` folder, these scripts test the performace of trained models. These evaluation scripts are currently not actively maintained and may require modifications to work with your setup.



## Supported Architectures
Our model trainer has a configurable backbone encoder, such as ResNet18. This backbone acts as a feature extractor. All architectures support both ABMIL and ITS2CLR-style training, with additional modifications based on their specific approach. Each architecture extends the base ABMIL and ITS2CLR capabilities with its own unique improvements and methodologies.

### Supported Encoders Families
- resnet
- convnextv2
- efficientnet
- efficientnet_v2

### ABMIL + ITS2CLR (Base architecture)
- ABMIL: https://arxiv.org/pdf/1802.04712.pdf
- ITS2CLR: https://arxiv.org/pdf/2210.09452.pdf
- ABMIL uses attention to determine the relations between parts of data.
- ITS2CLR structures the training to alternate between instance level and bag training, while sending sudo labels from bag level to instance level.

### Gen_ITS2CLR
- Source: https://arxiv.org/pdf/2106.00908.pdf

### PALM
- Source: https://arxiv.org/abs/2402.02653

### Rethinking MIL
- Source: https://arxiv.org/abs/2307.02249



## Lead Model
Our current direction is to include PALM in the instance training of our ABMIL + ITS2CLR framework. The bag training generates sudo labels that will be used in the instance training after warmup. PALM will move unknown instances to their closest prototype after warmup.



### Validation Results (WIP)
- Bag Acc: 83.72%
- Bag AUC: 90.75%
- Instance Acc: 67.32%
- Instance AUC: 74.05%
- Palm Acc: 64.61%

## Helpful Methods

These are helpful methods that are used to easily load and prepare systems for use. 
- `config = build_config(model_version, head_name, data_config)`
     - Loads config data from `config.py`
- `config = load_model_config(model_path)`
     - Loads config data from the model
- `bags_train, bags_val, bag_dl_train, bag_dl_val = prepare_all_data(config)`
     - Preloads all data and bag dataloaders 
- `model = build_model(config)`
     - Generates selected model architecture
- `model, optimizer, state = setup_model(model, config, optimizer = None)`
     - Generates or loads (if exists) current model state
- `palm.load_state(state['palm_path'])` or `palm.save_state(state['palm_path'])`
     - Uses model state to save/load PALM state
- `save_metrics(config, state, train_pred, val_pred)`
     - Evaluate and generate metrics of model performance
- `save_state(state, config, instance_train_acc, val_losses.avg, instance_val_acc, model, optimizer)`
     - Save current model state


## Test Dataset

We use a modified version of [ImageNet](https://paperswithcode.com/dataset/imagenet), specifically the ImageNette subset, to test our MIL architectures. The `./util/create_imagenet.py` script automatically generates this dataset by:

- Downloading the ImageNette dataset (a 10-class subset of ImageNet)
- Creating bags of 2-10 images each
- Generating both bag-level and instance-level labels
- Converting the data into our required format (TrainData.csv and InstanceData.csv)
- Organizing images into the expected directory structure

The script creates a binary classification problem where the target class is 'n01440764', with approximately 20% of bags containing positive instances. The dataset is automatically split with 80% for training and 20% for validation.

This synthetic dataset provides a controlled environment for testing MIL architectures before deploying them on medical imaging data.

## Dataloader Input Format

[CADBUSI-Database](https://github.com/Poofy1/CADBUSI-Database) will provide exports with the expected data format, if you need to use your own data you need to compile your data into this structured tabular format:

### Train_Data.csv

The file `/Export/Train_Data.csv` must exist with the following columns: `ID, Images, {User Labels}, Valid`. This describes each bag, what images and labels they have.
Example: 
```
ID,Images,Has_Malignant,Has_Benign,Valid
1,"['1_1_left_0.png', '1_1_left_6.png', '1_1_left_9.png']",True,False,0
```

#### 'ID'
- Type: Integer
- Description: A unique identifier for each record in the dataset.
#### 'Images'
- Type: String (formatted as a list of image file names)
- Description: Contains all the image file names associated with the bag. Each file name is a string within a list, formatted as 'image_name.png'. Multiple images are included for each ID.
#### User Labels (e.g., 'Has_Malignant', 'Has_Benign')
- Type: Boolean (True or False)
- Description: These columns represent various labels or annotations associated with each record. They are optional and can vary depending on the specific requirements of the study or analysis.
    - Example: Has_Malignant indicates whether malignant features are present in the images.
    - Users can include as many label columns as necessary for their analysis.
    - Users must include which labels they want to train on in each training script. Example: `label_columns = ['Has_Malignant', 'Has_Benign']`
#### 'Valid'
- Type: Integer (0, 1, or 2)
- Description: Indicates the usage of the record in the dataset.
- 0: Record is not part of the training or validation set.
- 1: Record is part of the training set.
- 2: Record is part of the test set.


### Image Folder

Inside the export folder should exist a `/Export/images/` to hold the images that are referenced in the `/Export/Train_Data.csv`.

### InstanceData.csv (Optional)

This file `/Export/InstanceData.csv` is optional but can provide labels for image instances for bags. If this is being used it must include these columns: `{User Labels}, ImageName`
Example: 
```
Benign Lesion Present, Malignant Lesion Present, ImageName
False, True, 2900_3081_left_0.png
```

#### User Labels (e.g., 'Has_Malignant', 'Has_Benign')
- Type: Boolean (True or False)
- Description: These columns represent various labels or annotations associated with each record. They are optional and can vary depending on the specific requirements of the study or analysis.
- Functions the same as the previous User Labels in `Train_Data.csv`

#### 'ImageName'
- Type: String
- Description: Contains a image file name associated with the label(s). 


## Data Pipeline
- [CADBUSI-Anonymize](https://github.com/Poofy1/CADBUSI-Anonymize)
- [CADBUSI-Database](https://github.com/Poofy1/CADBUSI-Database)
- [CADBUSI-Training](https://github.com/Poofy1/CADBUSI-Training)
![CASBUSI Pipeline](https://raw.githubusercontent.com/Poofy1/CADBUSI-Database/main/pipeline/CADBUSI-Pipeline.png)

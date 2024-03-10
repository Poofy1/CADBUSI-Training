# CADBUSI-Training

This project is a model training framework that supports various deep learning architectures for ultrasound image analysis. It's designed to be flexible and efficient, catering to a wide range of applications. This project is still a WIP.

## Requirements

- Python 3.8
- Nvidia GPU (Recommended)
- Install required Python packages with pip:

```
pip install -r requirements.txt
```



## Dataloader Input Format

[CADBUSI-Database](https://github.com/Poofy1/CADBUSI-Database) has formatted folder directory to the expected data format already, if you wanted to use your own data you would first need to build a script to compile your data into this structured tabular format. This is the expected data format:

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

This file `/Export/InstanceData.csv` is optional but can provide some labels for image instances for bags. If this is being used it must include these columns: `{User Labels}, ImageName`
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


## Usage

Once you have your input data formatted, you can begin training by running one of the provided training scripts in the immediate directory. Each script has its own config that you should configure given your task specifications. If you are running a training script on newly processed data, the script will crop and save the images into a temporary directory specified by you to improve performance. It will also manage unbalanced data by upsampling the minority class until they are balanced (This only works best when training on 1 label).



## Supported Architectures

At the core of our model trainer is a configurable backbone based on convolutional neural networks, such as ResNet18. This backbone acts as a feature extractor. This approach allows for the flexibility of using advanced pre-trained models while customizing the network's final layers to suit different requirements.

Our model trainer currently supports the following architectures:

### Gen_ITS2CLR (Custom)

- This uses a combination of different techniques given these two papers:
    - GenSCL: https://arxiv.org/pdf/2106.00908.pdf
    - ITS2CLR: https://arxiv.org/pdf/2210.09452.pdf
    - This architecture leverages generalized contrasitive learning alongside techniques used in ITS2CLR

### ABMIL (Attention-Based Multiple Instance Learning)

- Source: https://arxiv.org/pdf/1802.04712.pdf
- ABMIL is ideal for tasks where the relation between parts of the data is crucial.
- This architecture leverages attention mechanisms to focus on relevant parts of the input data.
- A Mixup version of this exists to try and prevent any overfitting.
- Results:
    - Failed to learn past 84% validation accuracy.
    - Feature maps failed to capture accurate feauture locations.
    - We believe this model is not able to learn the more difficult features. 

### FC

- Traditional FC networks with Resnet, used as a baseline.


## Data Pipeline
- [CADBUSI-Anonymize](https://github.com/Poofy1/CADBUSI-Anonymize)
- [CADBUSI-Database](https://github.com/Poofy1/CADBUSI-Database)
- [CADBUSI-Training](https://github.com/Poofy1/CADBUSI-Training)
![CASBUSI Pipeline](https://raw.githubusercontent.com/Poofy1/CADBUSI-Database/main/pipeline/CADBUSI-Pipeline.png)

# CADBUSI-Training

This project is a comprehensive model training framework that supports various deep learning architectures for image analysis. It's designed to be flexible and efficient, catering to a wide range of applications. This project is still a WIP.

## Requirements

- Python 3.8
- Nvidia GPU (Recommended)
- Install required Python packages with pip:

```
pip install -r requirements.txt
```


## Dataloader Input Format

[CADBUSI-Database](https://github.com/Poofy1/CADBUSI-Database) has formatted `Train_Data.csv` to the expected data format already, if you wanted to use different data you would need to build a script to compile your data into this structured tabular format with the following columns:
### ID
- Type: Integer
- Description: A unique identifier for each record in the dataset.
### Images
- Type: String (formatted as a list of image file names)
- Description: Contains all the image file names associated with the bag. Each file name is a string within a list, formatted as 'image_name.png'. Multiple images are included for each ID.
### Label Columns (e.g., Has_Malignant, Has_Benign)
- Type: Boolean (True or False)
- Description: These columns represent various labels or annotations associated with each record. They are optional and can vary depending on the specific requirements of the study or analysis.
    - Example: Has_Malignant indicates whether malignant features are present in the images.
    - Users can include as many label columns as necessary for their analysis.
    - Users must include which labels they want to train on in each training script. Example: `label_columns = ['Has_Malignant', 'Has_Benign']`
### Valid
- Type: Integer (0, 1, or 2)
- Description: Indicates the usage of the record in the dataset.
- 0: Record is not part of the training or validation set.
- 1: Record is part of the training set.
- 2: Record is part of the test set.

### Example
```
ID,Images,Has_Malignant,Has_Benign,Valid
1,"['1_1_left_0.png', '1_1_left_6.png', ...]",True,False,0
```

## Supported Architectures

At the core of our model trainer is a configurable backbone based on convolutional neural networks, such as ResNet18. This backbone acts as a feature extractor. This approach allows for the flexibility of using advanced pre-trained models while customizing the network's final layers to suit different requirements.

Our model trainer currently supports the following architectures:

### ABMIL (Attention-Based Multiple Instance Learning)

- Source: https://arxiv.org/pdf/1802.04712.pdf
- ABMIL is ideal for tasks where the relation between parts of the data is crucial.
- This architecture leverages attention mechanisms to focus on relevant parts of the input data.

### FC

- Traditional FC networks with Resnet.

### ITS2CLR

- Source: https://arxiv.org/pdf/2210.09452.pdf

### TransMIL (Transformer-Based Multiple Instance Learning)

- Source: https://arxiv.org/pdf/2106.00908.pdf

### GenSCL

- Source: https://arxiv.org/pdf/2106.00908.pdf


## Data Pipeline
- [CADBUSI-Anonymize](https://github.com/Poofy1/CADBUSI-Anonymize)
- [CADBUSI-Database](https://github.com/Poofy1/CADBUSI-Database)
- [CADBUSI-Training](https://github.com/Poofy1/CADBUSI-Training)
![CASBUSI Pipeline](https://raw.githubusercontent.com/Poofy1/CADBUSI-Database/main/pipeline/CADBUSI-Pipeline.png)

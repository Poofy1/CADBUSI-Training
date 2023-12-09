# CADBUSI-Training

This project is a comprehensive model training framework that supports various deep learning architectures for image analysis. It's designed to be flexible and efficient, catering to a wide range of applications. This project is still a WIP.

## Supported Architectures

At the core of our model trainer is a configurable backbone based on convolutional neural networks, such as ResNet18. This backbone acts as a feature extractor. This approach allows for the flexibility of using advanced pre-trained models while customizing the network's final layers to suit different requirements.

Our model trainer currently supports the following architectures:

### ABMIL (Attention-Based Multiple Instance Learning)

- Source: https://arxiv.org/pdf/1802.04712.pdf
- ABMIL is ideal for tasks where the relation between parts of the data is crucial.
- This architecture leverages attention mechanisms to focus on relevant parts of the input data.

### FC

- Traditional FC networks.
- They're particularly useful for tasks with well-defined, structured input data.

### ITS2CLR

- Source: https://arxiv.org/pdf/2210.09452.pdf

### TransMIL (Transformer-Based Multiple Instance Learning)

- Source: https://arxiv.org/pdf/2106.00908.pdf

### GenSCL

- Source: https://arxiv.org/pdf/2106.00908.pdf

## Data Formatting

Our framework supports universal data formatting, allowing for seamless integration and data preprocessing. This is the expected data formatting:
- A folder dir that holds the following:
    - `images` folder that holds all images
    - `TrainData.csv` that includes these headers:
      - "ID" (Bag ID)
      - "Images" (List of the image names that are in each bag)
        - Example: ['1_1_left_0.png', '1_1_left_2.png', '1_1_left_5.png', '1_1_left_7.png']
      - Valid (0 for training data, 1 for validation data)
      - Then any other header variables you want to train on, just make sure to specify them in the train scripts.

## Getting Started

- Python 3.8
- Nvidia GPU (Recommended)
- Install required Python packages with pip:

```
pip install -r requirements.txt
```

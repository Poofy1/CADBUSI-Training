# CASBUSI-Training

This project is a comprehensive model training framework that supports various deep learning architectures for image analysis. It's designed to be flexible and efficient, catering to a wide range of applications. This project is still a WIP.

## Supported Architectures

At the core of our model trainer is a configurable backbone based on convolutional neural networks, such as ResNet18. This backbone acts as a feature extractor. This approach allows for the flexibility of using advanced pre-trained models while customizing the network's final layers to suit different requirements.

Our model trainer currently supports the following architectures:

### ABMIL (Attention-Based Multiple Instance Learning)

- ABMIL is ideal for tasks where the relation between parts of the data is crucial.
- This architecture leverages attention mechanisms to focus on relevant parts of the input data.

### FC

- Traditional FC networks.
- They're particularly useful for tasks with well-defined, structured input data.

### TransMIL (Transformer-Based Multiple Instance Learning)

- TransMIL 

## Universal Data Formatting

Our framework supports universal data formatting, which allows for seamless integration and preprocessing of data. 

## Getting Started

- Python 3.8
- Nvidia GPU (Recommended)
- Install required Python packages with pip:

```
pip install -r requirements.txt
```
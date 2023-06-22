# Plant Disease Classification

This project is a deep learning based solution for classifying plant images by type and detecting diseases present in them, if any. It is built in Python using PyTorch for the neural network architecture and training process.

The system operates in two stages using a hierarchical method. Firstly, an image is classified to its plant type, and secondly, a disease-specific neural network corresponding to the identified plant type classifies the image to a disease class or healthy class.

## Methodology

- The architecture is structured to have multiple neural networks - one initial network for classifying the plant image into its type and then subsequent networks, each specific to a plant type, that further classify the image into disease categories or a healthy class.

- The plant classification model utilizes the pre-trained ResNet50 model from PyTorch's torchvision module. ResNet50 was chosen due to its excellent performance in handling images, which is central to our project.

## Requirements

- Python (3.7 or later)
- PyTorch (1.5.0 or later)
- Torchvision (0.6.0 or later)
- Numpy (1.18.5 or later)

## Dataset

The model is trained and validated on the PlantVillage Dataset, which includes colored images of healthy and diseased leaves from a variety of plant species. The `data_plant_type_consolidation.py` script is used to prepare this dataset for the training of the plant classifier neural network.

Currently, our model supports the following plants:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

## Usage

This project can be used as a standalone script, imported as a module, or integrated into a larger system. Please check the individual script files for more detailed usage instructions.


---

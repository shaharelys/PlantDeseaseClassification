# Plant Disease Classification

This project is a deep learning based solution for classifying plant images by type and detecting diseases present in them, if any. It is built in Python using PyTorch for the neural network architecture and training process.

The system operates in two stages using a hierarchical method. Firstly, an image is classified to its plant type, and secondly, a disease-specific neural network corresponding to the identified plant type classifies the image to a disease class or healthy class.

## Methodology

- The architecture is structured to have multiple neural networks - one initial network for classifying the plant image into its type and then subsequent networks, each specific to a plant type, that further classify the image into disease categories or a healthy class.

- The plant classification model utilizes the pre-trained ResNet50 model from PyTorch's torchvision module. The ResNet50 model has been widely used for image processing tasks due to its robust performance. The final layer of the ResNet50 model has been replaced to adapt the network for our specific number of plant type classes.

## Requirements

- Python (3.7 or later)
- PyTorch (1.5.0 or later)
- Torchvision (0.6.0 or later)
- Numpy (1.18.5 or later)
- PIL (7.1.2 or later)

Please note that the training process is resource-intensive. Using a free GPU on Google Colab, it takes a significant amount of time and utilizes about 90% of the available GPU RAM. It is recommended to use a more powerful GPU for faster training.

## Dataset

The model is trained and validated on the [colored images](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color) of the [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset/). These are colored images of healthy and diseased leaves from a variety of plant species. The `data_plant_type_consolidation.py` script is used to prepare this dataset for the training of the plant classifier neural network.

Our model supports the following plants found in PlantVillage:

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

---

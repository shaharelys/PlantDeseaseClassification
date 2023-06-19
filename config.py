# config.py

# model.py configurations
PLANT_CLASSES = ['Pepper_bell', 'Potato', 'Tomato']

# data configurations
TRAIN_RATIO = 0.7
BATCH_SIZE = 32
RESNET_1D_INPUT_SIZE = 224
R_MEAN, G_MEAN, B_MEAN = 0.485, 0.456, 0.406  # ImageNet's RGB means
R_STD, G_STD, B_STD = 0.229, 0.224, 0.225  # ImageNet's RGB std

# train.py configurations
NUM_EPOCH = 25

# main.py configurations
# DATA_DIR = "assets/PlantVillage_Plant_Classification"
DATA_DIR = "/content/drive/MyDrive/Plant_Classification/assets"
LEARNING_RATE = 0.001
MOMENTUM = 0.9
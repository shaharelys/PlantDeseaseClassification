# config.py

# model module configurations
PLANT_CLASSES = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato"
]

# data module  configurations
TRAIN_RATIO = 0.7
BATCH_SIZE = 128
RESNET_1D_INPUT_SIZE = 224
R_MEAN, G_MEAN, B_MEAN = 0.485, 0.456, 0.406  # ImageNet's RGB means
R_STD, G_STD, B_STD = 0.229, 0.224, 0.225  # ImageNet's RGB std

# train module configurations
NUM_EPOCH = 25
SAVE_INTERVAL = 5
WEIGHTS_FILE_PATH = "/content/drive/MyDrive/Plant_Classification/assets/weights"

# main module configurations
DATA_DIR_OLD = "/content/drive/MyDrive/Plant_Classification/assets/images/PlantVillage-Dataset/raw/color/"
DATA_DIR = "/content/drive/MyDrive/Plant_Classification/assets/images/PlantVillage-Dataset/raw/organized/"
LEARNING_RATE = 0.001
MOMENTUM = 0.9
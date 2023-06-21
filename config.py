# config.py

# model module configurations
PLANT_CLASSES = ['Apple_Consolidated',
                 'Blueberry_Consolidated',
                 'Cherry_(including_sour)_Consolidated',
                 'Corn_(maize)_Consolidated',
                 'Grape_Consolidated',
                 'Orange_Consolidated',
                 'Peach_Consolidated',
                 'Pepper,_bell_Consolidated',
                 'Potato_Consolidated',
                 'Raspberry_Consolidated',
                 'Soybean_Consolidated',
                 'Squash_Consolidated',
                 'Strawberry_Consolidated',
                 'Tomato_Consolidated']

# data module  configurations
TRAIN_RATIO = 0.7
BATCH_SIZE = 32
RESNET_1D_INPUT_SIZE = 224
R_MEAN, G_MEAN, B_MEAN = 0.485, 0.456, 0.406  # ImageNet's RGB means
R_STD, G_STD, B_STD = 0.229, 0.224, 0.225  # ImageNet's RGB std

# train module configurations
NUM_EPOCH = 25
SAVE_INTERVAL = 5
WEIGHTS_FILE_PATH = "/content/drive/MyDrive/Plant_Classification/assets/weights"

# main module configurations
DATA_DIR = "/content/drive/MyDrive/Plant_Classification/assets/images"
LEARNING_RATE = 0.001
MOMENTUM = 0.9
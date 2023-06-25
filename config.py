# config.py

# General configurations
ROOT_DIR = "/content/drive/MyDrive/Plant_Classification"
DATA_DIR_OLD = f"{ROOT_DIR}/assets/images/PlantVillage-Dataset/raw/color/"
DATA_DIR_1SNN = f"{ROOT_DIR}/assets/images/PlantVillage-Dataset/raw/organized/"
DATA_DIR_2SNNS = f"{ROOT_DIR}/assets/images/PlantVillage-Dataset/raw/2snns/"
WEIGHTS_FILE_PATH = f"{ROOT_DIR}/assets/weights"
WEIGHTS_FILE_PATH_2SNNS = f"{WEIGHTS_FILE_PATH}/weights_2snns"
WEIGHTS_FILE_PATH_1SNN = f"{WEIGHTS_FILE_PATH}/weights_1snn"

# model module configurations
PLANT_CLASSES = {
    "Apple": ["Apple_scab", "Black_rot", "Cedar_apple_rust", "healthy"],
    "Blueberry": ["healthy"],
    "Cherry": ["Powdery_mildew", "healthy"],
    "Corn": ["Cercospora_leaf_spot Gray_leaf_spot", "Common_rust", "Northern_Leaf_Blight", "healthy"],
    "Grape": ["Black_rot", "Esca_(Black_Measles)", "Leaf_blight_(Isariopsis_Leaf_Spot)", "healthy"],
    "Orange": ["Haunglongbing_(Citrus_greening)"],
    "Peach": ["Bacterial_spot", "healthy"],
    "Pepper": ["Bacterial_spot", "healthy"],
    "Potato": ["Early_blight", "Late_blight", "healthy"],
    "Raspberry": ["healthy"],
    "Soybean": ["healthy"],
    "Squash": ["Powdery_mildew"],
    "Strawberry": ["Leaf_scorch", "healthy"],
    "Tomato": ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "healthy"]
}

# data module configurations
SEED = 6061993  # picked my birthdate
TRAIN_RATIO = 3/100  # 3%
VALID_RATIO = TRAIN_RATIO/10  # 0.3%
TEST_RATIO = TRAIN_RATIO/10
DROPOUT_RATIO = 1 - TRAIN_RATIO - VALID_RATIO - TEST_RATIO  # not in actual use
BATCH_SIZE = 32  #128
RESNET_1D_INPUT_SIZE = 224
R_MEAN, G_MEAN, B_MEAN = 0.485, 0.456, 0.406  # ImageNet's RGB means
R_STD, G_STD, B_STD = 0.229, 0.224, 0.225  # ImageNet's RGB std

# train module configurations
NUM_EPOCH = 22
SAVE_INTERVAL_1SNN = 1
SAVE_INTERVAL_2SNN = 5
WEIGHT_FILE_PREFIX = "weights_epoch_"

# main module configurations
LEARNING_RATE = 0.001
MOMENTUM = 0.9
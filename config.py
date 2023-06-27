# config.py

# General configurations
ROOT_DIR = "/content/drive/MyDrive/python_projects/plant_disease_classification"
DATA_DIR_OLD = f"{ROOT_DIR}/assets/PlantVillage-Dataset/raw/color/"
DATA_DIR_1SNN = DATA_DIR_OLD  # f"{ROOT_DIR}/assets/PlantVillage-Dataset/raw/organized/"
DATA_DIR_2SNNS = f"{ROOT_DIR}/assets/PlantVillage-Dataset/raw/color/"
WEIGHTS_FILE_PATH = f"{ROOT_DIR}/assets/weights"
WEIGHTS_FILE_PATH_1SNN = f"{WEIGHTS_FILE_PATH}/1snn"
WEIGHTS_FILE_PATH_2SNNS = f"{WEIGHTS_FILE_PATH}/2snns"

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
    "Tomato": ["Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold", "Septoria_leaf_spot",
               "Spider_mites Two-spotted_spider_mite", "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus",
               "Tomato_mosaic_virus", "healthy"]
}
TOTAL_CLASSES_NUMBER = 38  # not in use in final code

# data module configurations
SEED = 6061993  # the seed for the random split (my birthdate)
TOTAL_USAGE_RATIO = 100/100  # 30% # not in use in final code
TRAIN_RATIO = 0.7*TOTAL_USAGE_RATIO
VALID_RATIO = 0.15*TOTAL_USAGE_RATIO
TEST_RATIO = 0.15*TOTAL_USAGE_RATIO
BATCH_SIZE = 128
RESNET_1D_INPUT_SIZE = 224
R_MEAN, G_MEAN, B_MEAN = 0.485, 0.456, 0.406  # ImageNet's RGB means
R_STD, G_STD, B_STD = 0.229, 0.224, 0.225  # ImageNet's RGB std

# train module configurations
NUM_EPOCH = 15
SAVE_INTERVAL_1SNN = 1
SAVE_INTERVAL_2SNN = 5
WEIGHT_FILE_PREFIX = "weights_epoch_"

# main module configurations
LEARNING_RATE = 0.001
MOMENTUM = 0.9
MODEL_NAMES_2SNN = []  # TODO
OPTIMIZER_PARAMS = {}

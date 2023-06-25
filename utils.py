# utils.py
import os
import re
import torch
from config import *


def get_device() -> torch.device:
    # Set the device for computation
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_weights(model, snn_type: str, plant_type: str = None) -> tuple[torch.nn.Module, int]:
    # Define weights_file_path based on snn_type and plant_type
    if snn_type == '1snn':
        weights_file_path = WEIGHTS_FILE_PATH_1SNN
    else:
        weights_file_path = f"{WEIGHTS_FILE_PATH_2SNNS}/{plant_type.lower()}"

    # Check if weight files exist and load weights from the file with the highest epoch
    last_epoch = None
    if os.path.exists(weights_file_path):
        weight_files = [f for f in os.listdir(weights_file_path) if f.endswith('.pth')]
        if weight_files:  # Check if the list is not empty
            weight_files.sort(key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1)))  # Sort files by epoch number
            weight_path = os.path.join(weights_file_path,
                                       weight_files[-1])  # Get the file with the highest epoch number
            model.load_state_dict(torch.load(weight_path))
            print(f'Loaded weights from file: {weight_path}')
            last_epoch = int(re.search(r'epoch_(\d+)', weight_files[-1]).group(1))

    return model, last_epoch

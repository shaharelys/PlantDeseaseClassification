# main.py
import torch
import torch.optim as optim
import torch.nn as nn
import os
import re
from model import model_plant_classifier
from data import create_data_loaders
from train import train_model, device
from config import *

# TODO: Add a README.md file

def main() -> None:

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(DATA_DIR)

    dataloaders = {'train': train_loader, 'val': valid_loader}

    # Move the model to the appropriate device (local variable renamed to avoid shadowing)
    plant_classifier_model = model_plant_classifier.to(device)

    # Check if weight files exist and load weights from the file with the highest epoch
    last_epoch = None
    if os.path.exists(WEIGHTS_FILE_PATH):
        weight_files = [f for f in os.listdir(WEIGHTS_FILE_PATH) if f.endswith('.pth')]
        if weight_files:  # Check if the list is not empty
            weight_files.sort(key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1)))  # Sort files by epoch number
            weight_path = os.path.join(WEIGHTS_FILE_PATH,
                                       weight_files[-1])  # Get the file with the highest epoch number
            plant_classifier_model.load_state_dict(torch.load(weight_path))
            print(f'Loaded weights from file: {weight_path}')
            last_epoch = int(re.search(r'epoch_(\d+)', weight_files[-1]).group(1))

    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(plant_classifier_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Train the model
    train_model(plant_classifier_model, dataloaders, criterion, optimizer, last_epoch)

    # TODO: Add here a test for the model

if __name__ == "__main__":
    main()

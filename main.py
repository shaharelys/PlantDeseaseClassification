# main.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import model_plant_classifier
from data import create_data_loaders
from utils import get_device, load_weights
from train import train_model
from evaluate import evaluate_model
from config import *


def main_1snn() -> None:
    # Set the device for computation
    device = get_device()

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(DATA_DIR_1SNN)
    dataloaders = {'train': train_loader, 'val': valid_loader}

    # Move the model to the appropriate device (local variable renamed to avoid shadowing)
    plant_classifier_model = model_plant_classifier.to(device)

    # Load the latest model weights, if exists
    plant_classifier_model, last_epoch = load_weights(plant_classifier_model, WEIGHTS_FILE_PATH)

    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(plant_classifier_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Train the model
    print('Training 1snn..')
    train_model(plant_classifier_model, dataloaders, criterion, optimizer, device, last_epoch)

    # Evaluate the model
    print('Evaluating 1snn..')
    evaluate_model(plant_classifier_model, test_loader, criterion, device)

"""

def main_2snns():
    # Train 2nd Stage Neural Networks (2snns)
    for model_name in MODEL_NAMES_2SNN:
        # Create data loader for the specific 2snn model
        train_loader, valid_loader, test_loader = create_dataloader_for_model(DATA_DIR_1SNN, model_name)
        dataloaders_2snn = {'train': train_loader, 'val': valid_loader}

        # Load the appropriate model
        model_2snn = load_model(model_name)

        # Load model weights if they exist
        model_2snn, last_epoch = load_model_weights(model_2snn, model_name)

        # Define the criterion
        criterion = nn.CrossEntropyLoss()

        # Define the optimizer for 2snn
        optimizer_params_2snn = OPTIMIZER_PARAMS[model_name]
        optimizer_2snn = optim.SGD(model_2snn.parameters(), **optimizer_params_2snn)

        # Train the 2snn model
        train_model(model_2snn, dataloaders_2snn, criterion, optimizer_2snn, DEVICE, last_epoch)

        # Evaluate the 2snn model
        evaluate_model(model_2snn, test_loader, criterion, DEVICE)

"""


if __name__ == "__main__":
    main_1snn()

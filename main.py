# main.py
import torch
import torch.optim as optim
import torch.nn as nn
from model import model_plant_classifier
from data import create_data_loaders
from train import train_model, device
from config import *


def main() -> None:

    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(DATA_DIR)

    dataloaders = {'train': train_loader, 'val': valid_loader}

    # Move the model to the appropriate device (local variable renamed to avoid shadowing)
    plant_classifier_model = model_plant_classifier.to(device)

    # Define the criterion and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(plant_classifier_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Train the model
    train_model(plant_classifier_model, dataloaders, criterion, optimizer)


if __name__ == "__main__":
    main()

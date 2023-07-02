# main.py
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
from config import *
from data import create_data_loaders
from utils import get_device
from model import load_model
from train import train_model
from evaluate import evaluate_model


def train_and_evaluate_model(device: torch.device,
                             dataloaders: dict[str, DataLoader],
                             model: nn.Module,
                             criterion: nn.Module,
                             optimizer: optim.Optimizer,
                             test_loader: DataLoader,
                             snn_type: str,
                             model_name: Optional[str] = None,
                             last_epoch: int = None,
                             save_interval: int = SAVE_INTERVAL_DEFAULT) -> None:

    # Train the model
    print(f'Training {model_name}..')
    train_model(model=model,
                device=device,
                dataloaders=dataloaders,
                criterion=criterion,
                optimizer=optimizer,
                snn_type=snn_type,
                model_name=model_name,
                last_epoch=last_epoch,
                save_interval=save_interval)

    # Evaluate the model
    print(f'Evaluating {model_name}..')
    evaluate_model(model=model,
                   dataloader=test_loader,
                   criterion=criterion,
                   device=device)


def main_1snn() -> None:
    # Acquire device for computation. If GPU is available, it uses GPU else it uses CPU.
    device = get_device()

    # Create data loaders for training, validation and testing datasets.
    train_loader, valid_loader, test_loader = create_data_loaders(representative_name='Full-snn1',
                                                                  data_dir=DATA_DIR_1SNN)

    # Combine train_loader and valid_loader in a dictionary for easier access during training and validation
    dataloaders = {'train': train_loader, 'val': valid_loader}

    # Load the model and check if there is any previous epoch saved for continuous training.
    # If yes, 'last_epoch' will have that epoch number, else it will be None.
    model, last_epoch = load_model('1snn')

    # Move the model to the acquired device for computation.
    model.to(device)

    # Define the loss function to be used in training
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer for the training.
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Call the function to train and evaluate the model.
    train_and_evaluate_model(device=device,
                             dataloaders=dataloaders,
                             model=model,
                             criterion=criterion,
                             optimizer=optimizer,
                             test_loader=test_loader,
                             snn_type='1snn',
                             model_name=None,
                             last_epoch=last_epoch,
                             save_interval=SAVE_INTERVAL_1SNN)


def main_2snns() -> None:
    # Acquire device for computation. If GPU is available, it uses GPU else it uses CPU.
    device = get_device()

    # For each model in the 2SNN model set
    for model_name in MODEL_NAMES_2SNN:
        # Validate model name
        if model_name not in PLANT_CLASSES:
            raise ValueError(f"Invalid plant type: {model_name}")

        # Define the directory for the specific model's data
        model_data_dir = f'{DATA_DIR_2SNNS}/{model_name}'

        # Create data loaders for training, validation and testing datasets.
        train_loader, valid_loader, test_loader = create_data_loaders(representative_name=model_name,
                                                                      data_dir=model_data_dir)

        # Combine train_loader and valid_loader in a dictionary for easier access during training and validation
        dataloaders_2snn = {'train': train_loader, 'val': valid_loader}

        # Load the model and check if there is any previous epoch saved for continuous training.
        # If yes, 'last_epoch' will have that epoch number, else it will be None.
        model, last_epoch = load_model(snn_type='2snn', plant_type=model_name)

        # Move the model to the acquired device for computation.
        model.to(device)

        # Define the loss function to be used in training
        criterion = nn.CrossEntropyLoss()

        # Define the optimizer for the training, the parameters are loaded from the OPTIMIZER_PARAMS dictionary.
        optimizer_params_2snn = OPTIMIZER_PARAMS_2SNNS.get(model_name, {'lr': LEARNING_RATE, 'momentum': MOMENTUM})
        optimizer_2snn = optim.SGD(model.parameters(), **optimizer_params_2snn)

        # Call the function to train and evaluate the model.
        train_and_evaluate_model(device=device,
                                 dataloaders=dataloaders_2snn,
                                 model=model,
                                 criterion=criterion,
                                 optimizer=optimizer_2snn,
                                 test_loader=test_loader,
                                 snn_type='2snn',
                                 model_name=model_name,
                                 last_epoch=last_epoch,
                                 save_interval=SAVE_INTERVAL_2SNN)


if __name__ == "__main__":
    # main_1snn()
    main_2snns()

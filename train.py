# train.py
import os
import torch
from typing import Dict, Optional
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from config import *

"""
def train_model_old(model: torch.nn.Module,
                device: torch.device,
                dataloaders: Dict[str, DataLoader],
                criterion: Module,
                optimizer: Optimizer,
                snn_type: str,
                plant_type: Optional[str] = None,
                last_epoch: Optional[int] = None,
                num_epochs: int = NUM_EPOCH,
                save_interval: int = SAVE_INTERVAL_DEFAULT
                ) -> None:

    start_epoch = 0 if last_epoch is None else last_epoch + 1
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # Save the model weights at the specified intervals
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            if snn_type == '1snn':
                weights_file_path = WEIGHTS_FILE_PATH_1SNN
            else:
                weights_file_path = f"{WEIGHTS_FILE_PATH_2SNNS}/{plant_type.lower()}"

            temp_path = f'{weights_file_path}/{WEIGHT_FILE_PREFIX}{epoch}.pth'
            torch.save(model.state_dict(), temp_path)
            print(f'---> New weights were successfully saved: {temp_path}')
"""

def calculate_loss_acc(model: Module,
                       dataloader: DataLoader,
                       criterion: Module,
                       optimizer: Optimizer,
                       phase: str,
                       device: torch.device
                       ) -> tuple[float, float]:

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc


def save_model_weights(model: Module,
                       weights_file_path: str,
                       epoch: int
                       ) -> None:
    temp_path = f'{weights_file_path}/{WEIGHT_FILE_PREFIX}{epoch}.pth'
    torch.save(model.state_dict(), temp_path)
    print(f'---> New weights were successfully saved: {temp_path}')


def save_model_weights_interval(model: Module,
                                snn_type: str,
                                epoch: int,
                                num_epochs: int,
                                plant_type: Optional[str] = None,
                                save_interval: int = SAVE_INTERVAL_DEFAULT
                                ) -> None:
    if epoch % save_interval == 0 or epoch == num_epochs - 1:
        if snn_type == '1snn':
            weights_file_path = WEIGHTS_FILE_PATH_1SNN
        else:
            weights_file_path = f"{WEIGHTS_FILE_PATH_2SNNS}/{plant_type.lower()}"

        # Create the directory if it doesn't exist
        if not os.path.exists(weights_file_path):
            os.makedirs(weights_file_path)

        save_model_weights(model, weights_file_path, epoch)


def train_model(model: torch.nn.Module,
                device: torch.device,
                dataloaders: Dict[str, DataLoader],
                criterion: Module,
                optimizer: Optimizer,
                snn_type: str,
                plant_type: Optional[str] = None,
                last_epoch: Optional[int] = None,
                num_epochs: int = NUM_EPOCH,
                save_interval: int = SAVE_INTERVAL_DEFAULT
                ) -> None:

    start_epoch = 0 if last_epoch is None else last_epoch + 1
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss, epoch_acc = calculate_loss_acc(model, dataloaders[phase], criterion, optimizer, phase, device)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        save_model_weights_interval(model, snn_type, epoch, num_epochs, plant_type, save_interval)

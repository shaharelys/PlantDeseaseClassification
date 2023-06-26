# data.py
import torch
import os
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from config import *


def create_data_loaders(data_dir: str,
                        train_ratio: float = TRAIN_RATIO,
                        valid_ratio: float = VALID_RATIO,
                        test_ratio: float = TEST_RATIO,
                        batch_size: int = BATCH_SIZE
                        ) -> tuple[DataLoader, DataLoader, DataLoader]:

    # Define transformations
    data_transforms = transforms.Compose([
        transforms.Resize((RESNET_1D_INPUT_SIZE, RESNET_1D_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[R_MEAN, G_MEAN, B_MEAN], std=[R_STD, G_STD, B_STD])
    ])

    # Load the dataset from the file system
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)

    # Split the dataset into training, validation, and test sets
    train_size = int(train_ratio * len(full_dataset))
    valid_size = int(valid_ratio * len(full_dataset))
    test_size = int(test_ratio * len(full_dataset))
    dropout_size = len(full_dataset) - train_size - valid_size - test_size  # the remaining size
    assert train_size + valid_size + test_size + dropout_size == len(
        full_dataset), "The sizes of the splits do not add up to the size of the full dataset."

    print(f'Dataset usage:\n'
          f'{train_size}\t instances were allocated for training.\n'
          f'{valid_size}\t instances were allocated for validation.\n'
          f'{test_size}\t instances were allocated for testing.\n'
          f'{dropout_size}\t instances were dropped out.\n')

    torch.manual_seed(SEED)
    train_dataset, valid_dataset, test_dataset, dropout_dataset = random_split(full_dataset,
                                                                               [train_size,
                                                                                valid_size,
                                                                                test_size,
                                                                                dropout_size])

    # Create DataLoaders for the training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader


def rearrange_data_1snn(base_dir: str = DATA_DIR_OLD, new_dir: str = DATA_DIR_1SNN) -> None:
    """
    Rearrange dataset in a new directory structure for the training of 1snn.
    """
    # List of plants
    plants = list(PLANT_CLASSES.keys())

    # Create directories for each plant
    for plant in plants:
        dir_path = os.path.join(new_dir, plant)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Print the contents of the base directory
    print(f"Contents of base directory {base_dir}:")
    print(os.listdir(base_dir))

    # Move files to the corresponding directory
    for dir_name in os.listdir(base_dir):
        print(f'Working on {dir_name}..')
        for plant in plants:
            # If the plant name is in the directory name
            if plant in dir_name:
                print(f'Noticed it is a {plant} dir..')
                # Full path to the directory
                dir_path = os.path.join(base_dir, dir_name)
                # If it is a directory
                if os.path.isdir(dir_path):
                    # Move all files in the directory to the corresponding plant directory
                    for file_name in os.listdir(dir_path):
                        old_file_path = os.path.join(dir_path, file_name)
                        new_file_path = os.path.join(new_dir, plant, file_name)
                        shutil.move(old_file_path, new_file_path)
                        print(f"Moved file from {old_file_path} to {new_file_path}")
                    # Remove the directory
                    os.rmdir(dir_path)
                    print(f"Deleted directory: {dir_path}")
                    break  # no need to check the other plants

# data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple
from config import *


def create_data_loaders(data_dir: str,
                        train_ratio: float = TRAIN_RATIO,
                        valid_ratio: float = VALID_RATIO,
                        test_ratio: float = TEST_RATIO,
                        batch_size: int = BATCH_SIZE
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:

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

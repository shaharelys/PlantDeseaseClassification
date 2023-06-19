# data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import *

# TODO: Add type hinting
def create_data_loaders(data_dir, train_ratio=TRAIN_RATIO, batch_size=BATCH_SIZE):
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
    valid_size = int((len(full_dataset) - train_size) / 2)
    test_size = len(full_dataset) - train_size - valid_size  # the remaining size
    train_dataset, valid_dataset, test_dataset = random_split(full_dataset, [train_size, valid_size, test_size])

    # Create DataLoaders for the training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader

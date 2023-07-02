# model.py
import torch
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from typing import Optional
from utils import load_weights
from config import *


def load_model(snn_type: str,
               plant_type: Optional[str] = None
               ) -> tuple[torch.nn.Module, int] or tuple[None, None]:
    """
    This function either returns a 1snn or a 2snn model based on the snn_type argument.
    For a 2snn, it also requires the plant_type argument.
    """

    # Validate input values and handle errors
    if snn_type not in ['1snn', '2snn']:
        raise ValueError(f"Invalid SNN type. Expected '1snn' or '2snn', got {snn_type}")

    if snn_type == '2snn':
        if plant_type is None:
            raise ValueError("Plant type must be specified when loading a 2snn model.")
        elif plant_type not in PLANT_CLASSES:
            raise ValueError(f"Invalid plant type: {plant_type}")

    # Define the number of output nodes based on snn type
    if snn_type == '1snn':
        num_classes = TOTAL_CLASSES_NUMBER  # len(PLANT_CLASSES)
    else:
        num_classes = len(PLANT_CLASSES[plant_type])

        if num_classes < 2:
            print(
                f"Warning: Insufficient disease classes found for the plant type: {plant_type}.\n"
                f"A model cannot be trained with less than two classes.\n\n")

            return None, None

    # Load pre-trained ResNet50
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Replace the final layer
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    # Load model weights if a training on Plant-Village has already occurred
    model, last_epoch = load_weights(model, snn_type, plant_type)

    return model, last_epoch



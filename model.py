# model.py
import torch
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from config import *

# Load pre-trained ResNet50
model_plant_classifier = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Replace the final layer
num_features = model_plant_classifier.fc.in_features
model_plant_classifier.fc = torch.nn.Linear(num_features, len(PLANT_CLASSES))

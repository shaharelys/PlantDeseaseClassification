# evaluate.py
import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def evaluate_model(model: Module,
                   dataloader: DataLoader,
                   criterion: Module,
                   device: torch.device) -> None:
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))

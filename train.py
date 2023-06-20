# train.py
import torch
from config import *

# Set the device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO: Add type hinting
def train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCH, save_interval=SAVE_INTERVAL):
    for epoch in range(num_epochs):
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
            temp_path = f'{WEIGHTS_FILE_PATH}/model_weights_epoch_{epoch}.pth'
            torch.save(model.state_dict(), temp_path)
            print(f'New weights were successfully saved: {temp_path}')


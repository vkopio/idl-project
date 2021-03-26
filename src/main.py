import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

from data_set import ImageDataSet
from model import CNN

CLASS_COUNT = 14
EPOCH_COUNT = 2
BATCH_SIZE = 100
LEARNING_RATE = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(CLASS_COUNT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss()

transformers = [
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5, inplace=True),
]

data_set = ImageDataSet('data/images', transforms.Compose(transformers))
train_set, val_set = data.random_split(data_set, [15000, 5000], generator=torch.Generator().manual_seed(42))
train_loader = data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

val_loader = data.DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)

def prediction_accuracy(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy() / len(original)

def evaluate(model, iterator, criterion):
    epoch_loss = []
    epoch_acc = []

    model.eval()
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(iterator):
            data, target = data.to(device), target.to(device)

            batch_prediction = model(data)
            loss = criterion(batch_prediction, target)

            batch_accuracy = []

            for i, prediction in enumerate(batch_prediction, 0):
                accuracy = prediction_accuracy(target[i], prediction)
                batch_accuracy.append(accuracy)

            epoch_loss.append(loss.data.item())
            epoch_acc.append(np.asarray(batch_accuracy).mean())

    return epoch_loss, epoch_acc

def train():
    for epoch_index in range(EPOCH_COUNT):
        total = len(train_loader.dataset)
        epoch_loss = []
        epoch_accuracy = []
        model.train()

        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            batch_prediction = model(data)
            loss = loss_function(batch_prediction, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_accuracy = []

            for i, prediction in enumerate(batch_prediction, 0):
                accuracy = prediction_accuracy(target[i], prediction)
                batch_accuracy.append(accuracy)

            epoch_loss.append(loss.data.item())
            epoch_accuracy.append(np.asarray(batch_accuracy).mean())

            print('Epoch %d | batch %d %% done' %
                  (epoch_index + 1,
                   100 * (batch_index + 1) / len(train_loader),),
                  end="\r",
                  flush=True)
            

        print('Epoch %d | training acc %.4f, loss %.4f ' %
              (epoch_index + 1,
               np.asarray(epoch_accuracy).mean(),
               np.asarray(epoch_loss).mean()))

        valid_loss, valid_acc = evaluate(model, val_loader, loss_function)

        print('Epoch %d | validation acc %.4f, loss %.4f ' %
              (epoch_index + 1,
               np.asarray(valid_acc).mean(),
               np.asarray(valid_loss).mean()))


if __name__ == '__main__':
    train()
    print("ok")

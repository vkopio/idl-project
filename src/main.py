# Import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

# Import model and data_set
from data_set import train_set, val_set, test_set
from model import CNN

# MacOs requires these two lines
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Define parameters
CLASS_COUNT = 14
EPOCH_COUNT = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_WORKERS = 10
RANDOM_SEED = 42

# Start training from beginning or continue with the trained model. True or False.
start_beginning=False

# Define path to trained model
model_dir = "trained_model.pth"

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get model
model = CNN(CLASS_COUNT).to(device)

# Define optimizer and loss_function ("criterion")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-5)

# Weights are defined to each class with formula: count_neg/count_pos.
#loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([209.5, 54.5, 61.7, 17.3, 43.6, 5.2, 25.3, 5.7, 32.4, 2.1, 5.4, 165.7, 114.6, 37.1]))
loss_function = torch.nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([106, 27,  31, 8, 22, 2, 12, 2, 16, 0.6, 2, 83, 57, 18]))

# Train, val and test loaders
train_loader = data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

val_loader = data.DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

test_loader = data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

# Initialize model weights (used when start training from beginning)
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)

def prediction_accuracy(original, predicted):
    original = original.to("cpu")
    predicted = predicted.to("cpu")
    return torch.round(predicted).eq(original).sum().numpy() / len(original)

def evaluate(model, iterator, criterion):
    epoch_loss = []
    epoch_acc = []

    model.eval()
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(iterator):
            data, target = data.to(device), target.to(device)
            batch_prediction = model(data)
            loss = criterion(batch_prediction.to("cpu"), target.to("cpu"))

            batch_accuracy = []

            for i, prediction in enumerate(batch_prediction, 0):
                accuracy = prediction_accuracy(target[i].to("cpu"), prediction.to("cpu"))
                batch_accuracy.append(accuracy)
            epoch_loss.append(loss.data.item())
            epoch_acc.append(np.asarray(batch_accuracy).mean())
    return epoch_loss, epoch_acc

def initialize_model(model, model_dir, device, start_beginning):
    # Initialize model weights
    model = model.apply(init_weights)
    # Load trained model
    if start_beginning == False:
        if device.type == "cpu":
            print("No Cuda available, load pretrained model to CPU")
            checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
            #optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])

        else:
            print("Load trained model to Cuda GPU")
            checkpoint = torch.load(model_dir)
            #optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])

    # Get starting validation accuracy and loss
    valid_loss, valid_acc = evaluate(model, val_loader, loss_function)
    validation_loss = np.asarray(valid_loss).mean()
    validation_acc = np.asarray(valid_acc).mean()
    print('Starting results | validation acc %.4f, loss %.4f ' %
              (validation_acc, validation_loss))
    return model, validation_acc

def train(model):
    # Initialize model
    model, best_valid_acc = initialize_model(model, model_dir, device, start_beginning)
    # Scheduler reduces learning rate after every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch_index in range(EPOCH_COUNT):
        total = len(train_loader.dataset)
        epoch_loss = []
        epoch_accuracy = []
        model.train()
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            batch_prediction = model(data)
            loss = loss_function(batch_prediction.to("cpu"), target.to("cpu"))
            loss.backward()
            optimizer.step()
            batch_accuracy = []

            for i, prediction in enumerate(batch_prediction, 0):
                accuracy = prediction_accuracy(target[i], prediction)
                batch_accuracy.append(accuracy)

            epoch_loss.append(loss.data.item())
            epoch_accuracy.append(np.asarray(batch_accuracy).mean())

            # Print epoch running progress
            print('Epoch %d | batch %d %% done' %
                  (epoch_index + 1,
                   100 * (batch_index + 1) / len(train_loader),),
                  end="\r", flush=True)
        
        scheduler.step()

        # Print training acc and loss
        print('Epoch %d | training acc %.4f, loss %.4f ' %
              (epoch_index + 1,
               np.asarray(epoch_accuracy).mean(),
               np.asarray(epoch_loss).mean()))

        # Print validation acc and loss
        valid_loss, valid_acc = evaluate(model, val_loader, loss_function)
        print('Epoch %d | validation acc %.4f, loss %.4f ' %
              (epoch_index + 1,
               np.asarray(valid_acc).mean(),
               np.asarray(valid_loss).mean()))

        # Save model if validation accuracy is better than best known one
        valid_acc = np.asarray(valid_acc).mean()
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            test_loss, test_acc = evaluate(model, test_loader, loss_function)
            print("New best result | Test accuracy: %.4f | Test loss %.4f" %
                (np.asarray(test_acc).mean(), np.asarray(test_loss).mean()))
            # Save model and weights used in further training and prediction
            state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, model_dir)
            print("New best model saved!")


if __name__ == '__main__':
    train(model)
    print("ok")

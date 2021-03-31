from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
import torch.utils.data as data
from typing import Union, List, Dict, Any, cast
import torchvision.models as models
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

#from data_set import ImageDataSet
from data_set import train_set, val_set, test_set

plt.ion()  
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

# MacOs requires these two lines
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Initialize parameters
CLASS_COUNT = 14
EPOCH_COUNT = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.01
NUM_WORKERS = 10
RANDOM_SEED = 42

# Start training from beginning or continue with the trained model. True or False.
start_beginning=True

# Define path to trained model
model_dir = "trained_vgg16_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adjust loss function weights!!
if use_gpu:
    loss_function = torch.nn.MultiLabelSoftMarginLoss(weight = torch.Tensor([1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1])).cuda()
else:
    loss_function = torch.nn.MultiLabelSoftMarginLoss(weight = torch.Tensor([1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1]))


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


def prediction_accuracy(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy() / len(original)

# Works!
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

# Works!
def initialize_VGG16(model_dir, device, start_beginning):
    # Load the pretrained model from pytorch
    vgg16 = models.vgg16(pretrained=True)
    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False
    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    # Get last layer
    features = list(vgg16.classifier.children())[:-1] 
    # Modify last layer to have 14 outputs
    features.extend([nn.Linear(num_features, CLASS_COUNT)]) 
    # Add new last layer to model
    vgg16.classifier = nn.Sequential(*features) 
    # Define optimizer
    optimizer = optim.SGD(vgg16.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # Load trained model. KESKEN!!!
    if start_beginning == False:
        if device.type == "cpu":
            print("No Cuda available, load pretrained model to CPU")
            checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("Load trained model to Cuda GPU")
            checkpoint = torch.load(model_dir)
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])


    # vgg16 to device
    vgg16 = vgg16.to(device)
    # Get starting validation accuracy and loss
    valid_loss, valid_acc = evaluate(vgg16, val_loader, loss_function)
    validation_loss = np.asarray(valid_loss).mean()
    print('Starting results | validation acc %.4f, loss %.4f ' %
              (np.asarray(valid_acc).mean(),
               validation_loss))
    # Return model and initial best valid loss
    return vgg16, valid_loss, optimizer


def train(model, best_valid_loss, optimizer):
    # Scheduler reduces learning rate after every 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch_index in range(EPOCH_COUNT):
        total = len(train_loader.dataset)
        epoch_loss = []
        epoch_accuracy = []
        model.train()
        # Loop through epochs
        for batch_index, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            batch_prediction = model(data)
            print(batch_prediction)
            loss = loss_function(batch_prediction, target)
            loss.backward()
            optimizer.step()
            batch_accuracy = []
            # Get accuracies
            for i, prediction in enumerate(batch_prediction, 0):
                accuracy = prediction_accuracy(target[i], prediction)
                batch_accuracy.append(accuracy)
            # Save to lists
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
        valid_loss = np.asarray(valid_loss).mean()
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            test_loss, test_acc = evaluate(model, test_loader, loss_function)
            print("New best result | Test accuracy: %.4f | Test loss %.4f" %
                (np.asarray(test_acc).mean(), np.asarray(test_loss).mean()))
            # Save model and weights used in further training and prediction
            state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, model_dir)
            print("New best model saved!")


if __name__ == '__main__':
    # Initialize model
    model, best_valid_loss, optimizer = initialize_VGG16(model_dir, device, start_beginning)

    # Train model
    train(model, best_valid_loss, optimizer)

    print("ok")

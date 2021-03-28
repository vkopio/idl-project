import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

#from data_set import ImageDataSet
from data_set import train_set, val_set, test_set
from model import CNN

# MacOs requires these two lines
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

CLASS_COUNT = 14
EPOCH_COUNT = 1
BATCH_SIZE = 100
LEARNING_RATE = 0.01
NUM_WORKERS = 10
RANDOM_SEED = 42

# Define path to trained model
model_dir = "trained_model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(CLASS_COUNT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCELoss()
#loss_function = nn.BCEWithLogitsLoss()

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

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)

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
                #print("Orig", target[i])
                #print("Prediction", prediction)
                accuracy = prediction_accuracy(target[i], prediction)
                batch_accuracy.append(accuracy)

            epoch_loss.append(loss.data.item())
            epoch_acc.append(np.asarray(batch_accuracy).mean())

    return epoch_loss, epoch_acc

def load_trained_model():
    # Load model
    checkpoint = torch.load(model_dir)
    '''
    if device.type == "cpu":
        print("No Cuda available, load pretrained model to CPU")
        #optimizer.load_state_dict(checkpoint['optimizer'], map_location='cpu')
        #model.load_state_dict(checkpoint['state_dict'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Load trained model to Cuda GPU")
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
    '''
    # Get starting validation accuracy and loss
    valid_loss, valid_acc = evaluate(model, val_loader, loss_function)
    validation_loss = np.asarray(valid_loss).mean()
    print('Starting results | validation acc %.4f, loss %.4f ' %
              (np.asarray(valid_acc).mean(),
               validation_loss))
    return validation_loss

def train(model):
    #model = CNN(CLASS_COUNT).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Load empty model with init weights. 
    # Use if you want to start training from beginning, and then comment load_trained_model().
    model = model.apply(init_weights)
    best_valid_loss = 100

    # Load trained model and init valiadation accuracy
    best_valid_loss = load_trained_model()
    

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

            # Print epoch running progress
            print('Epoch %d | batch %d %% done' %
                  (epoch_index + 1,
                   100 * (batch_index + 1) / len(train_loader),),
                  end="\r", flush=True)
        
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
    train(model)
    print("ok")

"""
Inspiration for the custom DataSet is drawn from:
https://discuss.pytorch.org/t/how-to-load-images-without-using-imagefolder/59999/3
"""

import os
import natsort
import numpy as np
from PIL import Image
from torch import FloatTensor
from torch.utils.data import Dataset
from torchvision import transforms
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from data_process import get_labels

image_dir = '../data/images'

labels = get_labels()
images = natsort.natsorted(os.listdir(image_dir))
image_indices = list(range(len(images)))

class ImageDataSet(Dataset):
    def __init__(self, image_indices, labels, transform):
        self.image_indices = image_indices
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        img_loc = os.path.join(image_dir, images[image_indices[index]])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        target = FloatTensor(self.labels[index])
        return tensor_image, target


def data_split(x, y, test_size=0.5, random_state=42):
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_index, test_index in msss.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test


img = np.array(image_indices)
lab = np.array(labels)

X_train, X_test_val, y_train, y_test_val = data_split(img, lab, test_size=0.25)

X_test, X_val, y_test, y_val = data_split(X_test_val, y_test_val, test_size=0.5)

transformers = [
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5, inplace=True),
]

train_set = ImageDataSet(X_train, y_train, transforms.Compose(transformers))
val_set = ImageDataSet(X_val, y_val, transforms.Compose(transformers))
test_set = ImageDataSet(X_test, y_test, transforms.Compose(transformers))

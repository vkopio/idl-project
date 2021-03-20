import os
import natsort
from PIL import Image
from torch import FloatTensor
from torch.utils.data import Dataset

from data_process import get_labels

labels = get_labels()

class ImageDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        target = FloatTensor(labels[idx])
        return tensor_image, target

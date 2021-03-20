import torch
import torch.utils.data as data
from torchvision import transforms

from data_set import ImageDataSet

BATCH_SIZE = 100


def main():
    transformers = [
        transforms.ToTensor(),
    ]

    data_set = ImageDataSet('data/images', transforms.Compose(transformers))

    train_loader = data.DataLoader(
        data_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )


if __name__ == '__main__':
    main()

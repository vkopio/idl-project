# Read libraries
from pathlib import Path
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
from itertools import compress

# Import model
from model import CNN

# Define paths
#weights_path = "trained_weights.pt" # Path to trained model weights
model_dir = "trained_model.pth"
read_dir = Path("../data/predict_images") # Directory to predict

# Labels list for printing
labels_list = ["baby", "bird", "car", "clouds", "dog", "female", "flower", "male", "night", "people", "portrait", "river", "sea", "tree"]

# Number of output classes
CLASS_COUNT = 14

# Image normalization
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std)

# The main
if __name__ == "__main__":
    # If available, use GPU, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model to device
    model = CNN(CLASS_COUNT).to(device)
    '''
    # Load trained model weights
    if device.type == "cpu":
        print("No Cuda available, will use CPU")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Use Cuda GPU")
        model.load_state_dict(torch.load(weights_path))
    '''
    # Load trained model weights
    if device.type == "cpu":
        print("No Cuda available, will use CPU")
        checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
        #model.load_state_dict(checkpoint['state_dict'], map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("Use Cuda GPU")
        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['state_dict'])

    # Get image paths
    img_paths = list(read_dir.glob("*.jpg"))
    img_paths = sorted(img_paths)

    # Loop images and make predictions
    for im_path in img_paths:
        # Image name for printing
        im_name = im_path.parts[-1]

        # Read image
        im = Image.open(im_path).convert("RGB") 
        im = torchvision.transforms.ToTensor()(im)

        # Normalize image
        normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        im = normalize(im)

        # Make prediction
        model.eval()
        with torch.no_grad():
            im = im.unsqueeze(0)
            prediction = model(im)


        # Prediction tensor to list
        prediction = prediction.numpy()[0]

        # Print results with values
        print(im_name, ":", prediction)

        # Binarize prediction. Threshold needs tuning!
        threshold = 0.155
        prediction[prediction < threshold] = 0
        prediction[prediction >= threshold] = 1

        # Print result with names
        labeled_prediction = list(compress(labels_list, prediction))
        print(im_name, ":", labeled_prediction)

        '''
        Correct answers for im1-im9
        im_num,count,baby,bird,car,clouds,dog,female,flower,male,night,people,portrait,river,sea,tree
        im1,     3,   0,    0,  0,   0,    0,   1,     0,    0,    0,    1,      1,      0,   0,  0
        im2,     0,   0,    0,  0,   0,    0,   0,     0,    0,    0,    0,      0,      0,   0,  0
        im3,     0,   0,    0,  0,   0,    0,   0,     0,    0,    0,    0,      0,      0,   0,  0
        im4,     2,   0,    0,  0,   0,    0,   0,     0,    1,    0,    1,      0,      0,   0,  0
        im5,     2,   0,    0,  0,   0,    0,   0,     0,    1,    0,    1,      0,      0,   0,  0
        im6,     0,   0,    0,  0,   0,    0,   0,     0,    0,    0,    0,      0,      0,   0,  0
        im7,     2,   0,    0,  0,   0,    0,   1,     0,    0,    0,    1,      0,      0,   0,  0
        im8,     0,   0,    0,  0,   0,    0,   0,     0,    0,    0,    0,      0,      0,   0,  0
        im9,     0,   0,    0,  0,   0,    0,   0,     0,    0,    0,    0,      0,      0,   0,  0
        '''
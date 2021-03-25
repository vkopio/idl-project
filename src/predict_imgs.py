# Read libraries
from pathlib import Path
from PIL import Image
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

# Import model
from model import CNN


# Define paths
weights_path = "trained_weights.pt" # Path to trained model weights
read_dir = Path("../data/predict_images") # Directory to predict


# Prediction main function
def predict(model, img_path, device):
    # Image to device
    img = img.to(device)
    # Make prediction
    for p in model.parameters():
        p.requires_grad = False
    # Set model to eval mode
    model.eval()
    # make prediction
    prediction = model(test_img)
    # Empty cuda caches. May be useless.
    torch.cuda.empty_cache()
    # Prediction detach
    if device.type == "cpu":
        prediction = prediction.detach().numpy()
    else:
        prediction = prediction.cpu().detach().numpy()
    # Get image name
    img_name = img_path.parts[-1]
    # Print result
    print(img_name)
    print(prediction)

# The main
if __name__ == "__main__":
    # If available, use GPU, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Model to device
    model = CNN.to(device)
    # Load trained model weights
    if device.type == "cpu":
        print("No Cuda available, will use CPU")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Use Cuda GPU")
        model.load_state_dict(torch.load(weights_path))

    # Get image paths
    img_paths = read_dir.glob("*.jpg")

    # Loop image paths and make predictions
    for img_path in img_paths:
        predict(model, img_path, device)
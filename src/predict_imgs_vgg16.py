# Read libraries
from pathlib import Path
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import functional as F
from itertools import compress
import torchvision.models as models
import torch.nn as nn

# Define paths
#weights_path = "trained_weights.pt" # Path to trained model weights
model_dir = "trained_vgg16_model.pth"
read_dir = Path("../data/predict_images") # Directory to predict

# Labels list for printing
labels_list = ["baby", "bird", "car", "clouds", "dog", "female", "flower", "male", "night", "people", "portrait", "river", "sea", "tree"]

# Number of output classes
CLASS_COUNT = 14

# Thresholds to turn prediction to labels. NEEDS TUNING!!!
thresholds = [0.8, 0.8, 0.8, 0.8, 0.8, 0.31, 0.8, 0.31, 0.8, 0.60, 0.30, 0.8, 0.8, 0.8]

# The main
if __name__ == "__main__":
    # If available, use GPU, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model to device
    model = models.vgg16(pretrained=True)

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    # Get last layer
    features = list(model.classifier.children())[:-1]
    # Modify last layer to have 14 outputs
    features.extend([nn.Linear(num_features, CLASS_COUNT)])
    # Add new last layer to model
    model.classifier = nn.Sequential(*features)

    # Load trained model
    if device.type == "cpu":
        print("No Cuda available, will use CPU")
        checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
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

        # To tensor and normalize
        im = transforms.Resize(224)(im)
        im = transforms.ToTensor()(im)
        im = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True)(im)

        # Make prediction
        model.eval()
        with torch.no_grad():
            im = im.unsqueeze(0)
            prediction = model(im)
        # Run sigmoid to prediction
        prediction = nn.functional.sigmoid(prediction)
        # Prediction tensor to list
        prediction = prediction.numpy()[0]

        # Print results with values
        print(im_name, ":", prediction)

        # Binarize prediction. Threshold needs tuning!
        for i in range(0, 14):
            if prediction[i] >= thresholds[i]:
                prediction[i] = 1
            else:
                prediction[i] = 0

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
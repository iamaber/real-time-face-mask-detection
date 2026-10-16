import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from config import Config, resolve_model_path

# Display labels in ImageFolder's sorted order: 0 -> "WithMask", 1 -> "WithoutMask".
CLASS_NAMES = ["With Mask", "Without Mask"]


def get_model(model_path=None, num_classes=2):
    config = Config()
    model_path = model_path or resolve_model_path()
    model = models.mobilenet_v3_large(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(p=config.dropout_rate),
        nn.Linear(1280, config.intermediate_features),
        nn.Hardswish(),
        nn.Dropout(p=config.dropout_rate),
        nn.Linear(config.intermediate_features, num_classes),
    )
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(config.device), weights_only=True)
    )
    model.to(config.device)
    model.eval()
    return model


def preprocess_image(image):
    """Convert an image to a normalised model input tensor.

    Accepts a PIL image or a numpy array. Arrays are assumed to be in the BGR
    order returned by ``cv2.imread`` and are converted to RGB; pass a PIL image
    when the data is already RGB.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if isinstance(image, np.ndarray):
        # cv2 decodes BGR, while the model was trained on RGB PIL images.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)


def predict(model, image_tensor):
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        # softmax to convert raw outputs to probabilities
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence_scores = {
            CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
        }

        return confidence_scores


def format_prediction(confidence_scores):
    """Format a prediction dict as a single label string for overlays."""
    label, score = max(confidence_scores.items(), key=lambda item: item[1])
    return f"{label}: {score:.2%}"

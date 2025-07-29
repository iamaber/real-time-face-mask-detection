import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Same Config class as in training script
class Config:
    def __init__(self):
        self.dropout_rate = 0.5
        self.intermediate_features = 640
        self.num_classes = 2  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_path, num_classes):
    config = Config()
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
    model.load_state_dict(torch.load(model_path, map_location=torch.device(config.device)))
    model.to(config.device)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.fromarray(image)
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def predict(model, image_tensor):
    class_names = ["With Mask", "Without Mask"]
    with torch.no_grad():
        outputs = model(image_tensor)
        # softmax to convert raw outputs to probabilities
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence_scores = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return confidence_scores
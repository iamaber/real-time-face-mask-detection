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

import cv2
from scripts.utils import get_model, preprocess_image, predict

def predict_image(image_path, model_path, num_classes=2):
    model = get_model(model_path, num_classes)
    image = cv2.imread(image_path)
    image_tensor = preprocess_image(image)
    prediction = predict(model, image_tensor)
    return prediction
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io
from scripts.utils import get_model, preprocess_image, predict

app = FastAPI()
model_path = "models/best_mobilenet_mask_detector.pt"
model = get_model(model_path, num_classes=2)

@app.post("/predict/")
async def predict_mask(file: UploadFile = File(...)):
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_np = np.array(image)
    image_tensor = preprocess_image(image_np)
    confidence_scores = predict(model, image_tensor)
    
    return confidence_scores
import gradio as gr
import requests
import numpy as np
import cv2

def predict_mask_image(image):
    
    _, image_encoded = cv2.imencode(".jpg", image) # convert image format to a file-like object for the API
    files = {"file": ("image.jpg", image_encoded.tobytes(), "image/jpeg")}

    response = requests.post("http://127.0.0.1:8000/predict/", files=files)
    
    return response.json()

def predict_mask_video(video_path):
    
    # For simplicity, we'll just return a placeholder for now.
    # A full implementation would process the video frame by frame.
    return "Video processing funtonality not implemented yet."
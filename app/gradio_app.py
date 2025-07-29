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

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Real-Time Mask Detector")
    with gr.Tab("Image"):
        image_input = gr.Image()
        image_output = gr.Label(label="Results")
        image_button = gr.Button("Predict")
    with gr.Tab("Video"):
        video_input = gr.Video()
        video_output = gr.Label()
        video_button = gr.Button("Predict")

    image_button.click(predict_mask_image, inputs=image_input, outputs=image_output)
    video_button.click(predict_mask_video, inputs=video_input, outputs=video_output)

if __name__ == "__main__":
    demo.launch()
    
import os
import threading
import time

import cv2
import gradio as gr
import requests
from main import app

API_HOST = os.environ.get("MASK_DETECTOR_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("MASK_DETECTOR_API_PORT", 8000))
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
GRADIO_HOST = os.environ.get("MASK_DETECTOR_GRADIO_HOST", "127.0.0.1")
REQUEST_TIMEOUT = 30


def predict_mask_image(image):
    _, image_encoded = cv2.imencode(
        ".jpg", image
    )  # convert image format to a file-like object for the API
    files = {"file": ("image.jpg", image_encoded.tobytes(), "image/jpeg")}

    response = requests.post(
        f"{API_BASE_URL}/predict/", files=files, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    return response.json()


def predict_mask_video(video_path):
    if video_path is None:
        return {"error": "No video file provided"}

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}

        frame_predictions = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames (process every 10th frame for efficiency) at most 50 frames
        sample_interval = max(1, total_frames // 50)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0:
                _, frame_encoded = cv2.imencode(".jpg", frame)
                files = {"file": ("frame.jpg", frame_encoded.tobytes(), "image/jpeg")}

                try:
                    response = requests.post(
                        f"{API_BASE_URL}/predict/",
                        files=files,
                        timeout=REQUEST_TIMEOUT,
                    )
                    if response.status_code == 200:
                        prediction = response.json()
                        frame_predictions.append(prediction)
                except requests.exceptions.RequestException:
                    pass

            frame_count += 1

        cap.release()

        if not frame_predictions:
            return {"error": "No predictions could be made on video frames"}

        with_mask_scores = [pred.get("With Mask", 0) for pred in frame_predictions]
        without_mask_scores = [
            pred.get("Without Mask", 0) for pred in frame_predictions
        ]

        avg_with_mask = sum(with_mask_scores) / len(with_mask_scores)
        avg_without_mask = sum(without_mask_scores) / len(without_mask_scores)

        overall_prediction = (
            "With Mask" if avg_with_mask > avg_without_mask else "Without Mask"
        )
        confidence = max(avg_with_mask, avg_without_mask)

        return {
            "Overall Prediction": overall_prediction,
            "Confidence": f"{confidence:.2%}",
            "With Mask (avg)": f"{avg_with_mask:.2%}",
            "Without Mask (avg)": f"{avg_without_mask:.2%}",
            "Frames Processed": len(frame_predictions),
            "Total Frames": total_frames,
        }

    except Exception as e:
        return {"error": f"Error processing video: {str(e)}"}


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Real-Time Mask Detector")
    gr.Markdown("Upload images or videos to detect whether people are wearing masks.")

    with gr.Tab("Image"):
        gr.Markdown("### Image Analysis")
        gr.Markdown("Upload an image to detect mask usage.")
        image_input = gr.Image()
        image_output = gr.Label(label="Results")
        image_button = gr.Button("Predict")

    with gr.Tab("Video"):
        gr.Markdown("### Video Analysis")
        gr.Markdown(
            "Upload a video to analyze mask usage across frames. The system will sample frames and provide aggregated results."
        )
        video_input = gr.Video()
        video_output = gr.JSON(label="Video Analysis Results")
        video_button = gr.Button("Predict")

    image_button.click(predict_mask_image, inputs=image_input, outputs=image_output)
    video_button.click(predict_mask_video, inputs=video_input, outputs=video_output)


def wait_for_api(timeout=60, interval=0.5):
    """Poll the API health endpoint until it responds instead of sleeping blindly."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(interval)
    raise RuntimeError(f"API did not become ready at {API_BASE_URL} within {timeout}s")


if __name__ == "__main__":
    import uvicorn

    # Start FastAPI server in a separate thread
    def run_fastapi():
        uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")

    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    wait_for_api()

    # Launch Gradio interface
    demo.launch(server_name=GRADIO_HOST)

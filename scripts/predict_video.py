import argparse

import cv2

from scripts.utils import format_prediction, get_model, predict, preprocess_image


def predict_video(video_source, model_path=None, num_classes=2):
    model = get_model(model_path, num_classes)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_tensor = preprocess_image(frame)
            prediction = predict(model, image_tensor)
            label = format_prediction(prediction)

            cv2.putText(
                frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            cv2.imshow("Mask Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Predict mask usage in a video.")
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to the input video, or a camera index such as 0",
    )
    parser.add_argument(
        "--model_path", default=None, help="Checkpoint path (defaults to the bundled model)"
    )
    args = parser.parse_args()

    source = int(args.video_path) if args.video_path.isdigit() else args.video_path
    predict_video(source, args.model_path)


if __name__ == "__main__":
    main()

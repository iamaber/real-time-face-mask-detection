import argparse

import cv2

from scripts.utils import format_prediction, get_model, predict, preprocess_image


def predict_image(image_path, model_path=None, num_classes=2):
    model = get_model(model_path, num_classes)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_tensor = preprocess_image(image)
    prediction = predict(model, image_tensor)
    return prediction


def main():
    parser = argparse.ArgumentParser(description="Predict mask usage in an image.")
    parser.add_argument("--image_path", required=True, help="Path to the input image")
    parser.add_argument(
        "--model_path", default=None, help="Checkpoint path (defaults to the bundled model)"
    )
    args = parser.parse_args()

    prediction = predict_image(args.image_path, args.model_path)
    for label, score in prediction.items():
        print(f"{label}: {score:.2%}")
    print(f"Prediction: {format_prediction(prediction)}")


if __name__ == "__main__":
    main()

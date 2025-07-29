import cv2
from scripts.utils import get_model, preprocess_image, predict

def predict_video(video_source, model_path, num_classes=2):
    model = get_model(model_path, num_classes)
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor = preprocess_image(frame)
        prediction = predict(model, image_tensor)

        cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Mask Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

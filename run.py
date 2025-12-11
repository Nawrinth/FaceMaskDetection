import cv2
import numpy as np
from tensorflow.keras.models import load_model


face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")


model = load_model("models/best_mask_detector.h5")
IMG_SIZE = 224  # CHANGE THIS TO YOUR MODEL INPUT SIZE


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Crop face ROI
        face_img = frame[y:y+h, x:x+w]

        # Preprocess ROI
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # Predict mask
        pred = model.predict(input_data)[0]

        if len(pred) == 1:
            label = "MASK" if pred[0] < 0.5 else "NO MASK"
        else:
            label = "MASK" if np.argmax(pred) == 0 else "NO MASK"

        # Box color
        color = (0, 255, 0) if label == "MASK" else (0, 0, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put text
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

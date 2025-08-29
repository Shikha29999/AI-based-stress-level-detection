import os
import cv2
import numpy as np
from keras.models import load_model

# Emotion categories used during model training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Stress mapping based on emotion
stress_mapping = {
    'Angry': 'High',
    'Disgust': 'High',
    'Fear': 'High',
    'Sad': 'Moderate',
    'Surprise': 'Moderate',
    'Neutral': 'Low',
    'Happy': 'Low'
}

# Load the trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
model = load_model(MODEL_PATH)

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(face_reshaped)
        max_index = int(np.argmax(prediction))
        predicted_emotion = emotion_labels[max_index]
        stress_level = stress_mapping[predicted_emotion]

        # Overlay results
        label = f'{predicted_emotion} ({stress_level} Stress)'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 100), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Live Stress Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

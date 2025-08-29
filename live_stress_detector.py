import os
import cv2
import numpy as np
from keras.models import load_model

# Emotion categories (7 classes used during training)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')
model = load_model(MODEL_PATH)

# Start video capture
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi]) != 0:
            roi = roi.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            prediction = model.predict(roi)[0]
            emotion = emotion_labels[np.argmax(prediction)]

            # Show emotion + stress level
            color = (0, 255, 0)
            label = f'{emotion}'
            if emotion in ['Happy', 'Surprise', 'Neutral']:
                stress_level = "Low Stress"
            elif emotion in ['Sad', 'Fear']:
                stress_level = "Moderate Stress"
            else:
                stress_level = "High Stress"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{emotion} - {stress_level}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Stress Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

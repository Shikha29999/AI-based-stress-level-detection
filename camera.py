import cv2
import numpy as np
from keras.models import load_model

# 1) लोड आपका CNN मॉडल एक बार
MODEL = load_model('model.h5')
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
STRESS = {
    'Angry': 'High', 'Disgust': 'High', 'Fear': 'High',
    'Sad': 'Moderate', 'Surprise': 'Moderate',
    'Neutral': 'Low', 'Happy': 'Low'
}

# 2) फेस डिटेक्शन कैस्केड
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 3) ग्लोबल वेरिएबल्स वर्तमान इमोशन/स्ट्रेस के लिए
CURRENT_EMOTION = 'Neutral'
CURRENT_STRESS = 'Low'

def get_current_status():
    """
    रूट /status में JSON भेजने के लिए उपयोग करें।
    """
    return CURRENT_EMOTION, CURRENT_STRESS

def gen_frames():
    """
    Flask को MJPEG फ्रेम्स yield करने के लिए।
    हर फ्रेम पर इमोशन और स्ट्रेस प्रिडिक्शन करके ओवरले भी करता है।
    """
    global CURRENT_EMOTION, CURRENT_STRESS

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # ग्रेस्केल में कन्वर्ट और फेस डिटेक्शन
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # ROI एक्सट्रैक्ट, रीसाइज़, नार्मलाइज, reshape
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))  # shape (1,48,48,1)

            # इमोशन प्रिडिक्शन
            pred = MODEL.predict(roi)[0]
            idx = np.argmax(pred)
            emotion = LABELS[idx]
            stress = STRESS[emotion]

            # ग्लोबल स्टेट अपडेट
            CURRENT_EMOTION = emotion
            CURRENT_STRESS = stress

            # ओवरले बॉक्स और टेक्स्ट
            label = f"{emotion} | {stress}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # JPEG में encode करके yield करो
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    cap.release()

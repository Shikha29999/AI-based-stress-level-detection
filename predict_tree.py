# app/predict_tree.py

import joblib
import numpy as np

# Load the correct model
model = joblib.load('remedy_tree_model.pkl')

def predict_stress(features: list):
    """
    Predict stress level using the decision tree.
    features = [emotion_code, sleep_hours, screen_time_hours, age]
    Returns: predicted stress level (e.g., Low, Moderate, High)
    """
    data = np.array([features])
    prediction = model.predict(data)
    return prediction[0]

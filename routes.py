from flask import Blueprint, Response, render_template, jsonify
from .camera import gen_frames, get_current_status

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@main.route('/detect')
def detect():
    return render_template('detect.html')

@main.route('/status')
def status():
    emotion, stress = get_current_status()
    return jsonify({'emotion': emotion, 'stress_level': stress})

@main.route('/result')
def result():
    return render_template('result.html')

@main.route('/remedies')
def remedies():
    # Страница с рекомендациями по снижению стресса
    return render_template('remedies.html')


@main.route('/predict_tree')
def predict_tree():
    import pickle
    import numpy as np

    # Example test input — replace with real data later
    test_data = np.array([[0, 1, 0]])  # Dummy feature input (adjust according to your model)
    
    # Load the model
    with open('remedy_tree_model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(test_data)[0]
    
    return render_template('tree_result.html', prediction=prediction)


@main.route('/decision_tree')
def decision_tree():
    return render_template('decision_tree.html')

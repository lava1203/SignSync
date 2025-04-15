from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import mediapipe as mp
import pyttsx3
import webbrowser
from threading import Timer, Thread

app = Flask(__name__)

# Configuration
SIGNS_FOLDER = 'static/signs_gif'
os.makedirs(SIGNS_FOLDER, exist_ok=True)

# Global variables
camera = None
model = None
label_encoder = None
current_prediction = ""
current_confidence = "0%"
sentence_mode = False
current_sentence = ""
engine = pyttsx3.init()

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)

# Load models
def load_models():
    global model, label_encoder
    try:
        model = load_model('lstm_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")

load_models()

# Text-to-Speech function
def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    Thread(target=_speak).start()

# Text-to-Sign Functionality
def text_to_sign_conversion(text):
    signs = []
    for word in text.lower().split():
        clean_word = ''.join(c for c in word if c.isalpha())
        if not clean_word:
            continue
            
        # Check for both .mp4 and .gif files
        video_file = None
        for ext in ['.mp4', '.gif']:
            if os.path.exists(os.path.join(SIGNS_FOLDER, f"{clean_word}{ext}")):
                video_file = f"{clean_word}{ext}"
                break
        
        signs.append({
            'word': clean_word,
            'filename': video_file,
            'available': video_file is not None
        })
    return signs

# Video feed generator for Sign-to-Text
def generate_frames():
    global current_prediction, current_confidence, current_sentence
    
    seq_buffer = []
    prediction_buffer = []
    prev_word = ""
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame (NO FLIPPING - ONLY CHANGE MADE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks[:2]:
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                while len(landmarks) < 126 * (results.multi_hand_landmarks.index(hand) + 1):
                    landmarks.extend([0.0, 0.0, 0.0])
        
        while len(landmarks) < 252:
            landmarks.extend([0.0, 0.0, 0.0])
        landmarks = landmarks[:252]
        
        seq_buffer.append(landmarks)
        if len(seq_buffer) > 25:
            seq_buffer.pop(0)
        
        # Prediction logic
        if len(seq_buffer) == 25 and model:
            prediction = model.predict(np.expand_dims(seq_buffer, axis=0), verbose=0)
            confidence = np.max(prediction)
            pred_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            current_confidence = f"{confidence*100:.1f}%"
            
            if confidence > 0.7:
                prediction_buffer.append(pred_class)
                if len(prediction_buffer) > 3:
                    prediction_buffer.pop(0)
                
                if len(prediction_buffer) == 3 and all(p == prediction_buffer[0] for p in prediction_buffer):
                    if pred_class != prev_word:
                        prev_word = pred_class
                        current_prediction = pred_class
                        speak(pred_class)
                        
                        if sentence_mode:
                            current_sentence += pred_class + " "
        
        # Draw UI elements
        cv2.putText(frame, f"Sign: {current_prediction}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {current_confidence}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if sentence_mode:
            cv2.putText(frame, "SENTENCE MODE: ON", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Sentence: {current_sentence[:30]}", (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text_to_sign', methods=['GET', 'POST'])
def text_to_sign_page():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if not text:
            return render_template('text_to_sign.html', 
                                error="Please enter text to convert",
                                input_text="")
        
        signs = text_to_sign_conversion(text)
        return render_template('text_to_sign.html',
                             signs=signs,
                             input_text=text)
    
    return render_template('text_to_sign.html', input_text="")

@app.route('/sign_to_text')
def sign_to_text_page():
    return render_template('sign_to_text.html')

@app.route('/video_feed')
def video_feed():
    global camera
    camera = cv2.VideoCapture(0)
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_sentence_mode', methods=['POST'])
def toggle_sentence_mode():
    global sentence_mode
    sentence_mode = not sentence_mode
    return jsonify({'status': 'success', 'sentence_mode': sentence_mode})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global current_sentence
    current_sentence = ""
    return jsonify({'status': 'success'})

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    data = request.get_json()
    speak(data['text'])
    return jsonify({'status': 'success'})

@app.route('/get_prediction')
def get_prediction():
    return jsonify({
        'prediction': current_prediction,
        'confidence': current_confidence,
        'sentence': current_sentence,
        'sentence_mode': sentence_mode
    })

@app.route('/stop_camera')
def stop_camera():
    global camera, current_prediction, current_confidence, current_sentence
    if camera is not None:
        camera.release()
    current_prediction = ""
    current_confidence = "0%"
    current_sentence = ""
    return jsonify({'status': 'success'})

def open_browser():
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True)
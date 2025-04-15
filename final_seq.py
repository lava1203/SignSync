import sys
import traceback

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("[❌] Uncaught exception:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import pickle
from collections import deque
from threading import Thread
import time

# Load model and encoder
model_start = time.time()
model = tf.keras.models.load_model("lstm_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
model_load_time = time.time() - model_start
print(f"[DEBUG] Model load time: {model_load_time:.2f}s")
print(f"[DEBUG] Encoded classes: {le.classes_}")  # Check if "okay" is in labels

# Text-to-speech
def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    Thread(target=_speak).start()

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[⚠️] Camera not accessible. Try cv2.VideoCapture(1) or check permissions.")
    exit()
seq_buffer = deque(maxlen=25)
prediction_buffer = deque(maxlen=3)
sentence = ""
prev_word = ""
sentence_mode = False
CONFIDENCE_THRESHOLD = 0.7  # Temporarily lowered to test "okay"

print("[🎥] SignSync Polished UI is running...")
print("[🕹️] Press 'm' to toggle Sentence Mode. Press 'q' to quit, 'c' to clear sentence, 's' to speak sentence.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[⚠️] Camera error.")
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mediapipe_start = time.time()

    try:
        results = hands.process(rgb)
    except Exception as e:
        print("[❌] MediaPipe crash avoided:", e)
        continue
    mediapipe_time = time.time() - mediapipe_start

    landmarks = []
    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)
        print(f"[DEBUG] {num_hands} hand(s) detected")
        for hand in results.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            while len(landmarks) < 126:
                landmarks.extend([0.0, 0.0, 0.0])
        while len(landmarks) < 252:
            landmarks.extend([0.0, 0.0, 0.0])
        landmarks = landmarks[:252]
        print(f"[DEBUG] Frame landmarks (first 10): {landmarks[:10]}")
        seq_buffer.append(landmarks)
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    else:
        print(f"[DEBUG] No hand detected")
        seq_buffer.append([0.0] * 252)

    if len(seq_buffer) == 25:
        predict_start = time.time()
        prediction = model.predict(np.expand_dims(list(seq_buffer), axis=0), verbose=0)
        prediction_time = time.time() - predict_start
        confidence = np.max(prediction)
        pred_class_idx = np.argmax(prediction)
        pred_class = le.inverse_transform([pred_class_idx])[0]
        total_time = time.time() - predict_start + mediapipe_time
        print(f"[DEBUG] MediaPipe time: {mediapipe_time:.2f}s, Prediction time: {prediction_time:.2f}s, Total: {total_time:.2f}s, Confidence: {confidence:.2f} for {pred_class}")
        if pred_class == "okay":
            print(f"[DEBUG] 'okay' detected with confidence: {confidence:.2f}")

        if confidence < 0.5:
            prediction_buffer.clear()
            seq_buffer.clear()
            print(f"[🔄] Reset buffers due to very low confidence")
        elif confidence > CONFIDENCE_THRESHOLD:
            prediction_buffer.append(pred_class)

        if len(prediction_buffer) == 3 and all(p == prediction_buffer[0] for p in prediction_buffer):
            if sentence_mode and pred_class != prev_word:
                sentence += pred_class + " "
                print(f"[🧠] Added to sentence: {pred_class}")
                prev_word = pred_class
                speak(pred_class)
            elif pred_class != prev_word:
                prev_word = pred_class
                speak(pred_class)
            prediction_buffer.clear()

    cv2.putText(frame, f"Detected: {prev_word.upper()}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, f"Sentence Mode: {'ON' if sentence_mode else 'OFF'} | Press 'm' to toggle",
                (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"Sentence: {sentence.strip()}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("SignSync Final UI", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        prev_word = ""
        print("[🧹] Sentence cleared.")
    elif key == ord('s'):
        print(f"[🔊] Speaking: {sentence.strip()}")
        speak(sentence.strip())
    elif key == ord('m'):
        sentence_mode = not sentence_mode
        print(f"[🔁] Sentence mode {'ON' if sentence_mode else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
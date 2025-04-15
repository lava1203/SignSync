import cv2
import numpy as np
import mediapipe as mp
import os
import time
import traceback

# Get gesture label and parameters
gesture = input("Enter gesture label (e.g., Hello): ").lower()
num_sequences = 50  # 50 sequences per sign
frames_per_seq = 25  # 25 frames per sequence

save_dir = f"seq_data_v2/{gesture}"
os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[⚠️] Camera not accessible. Try cv2.VideoCapture(1) or check permissions.")
    exit()
existing = len(os.listdir(save_dir))

print(f"[🎥] Starting collection of {num_sequences} sequences for '{gesture}'")

for seq_num in range(num_sequences):
    try:
        sequence = []
        print(f"\n[⏳] Starting sequence {seq_num + 1}/{num_sequences} in 2 seconds...")
        time.sleep(2)

        frames_collected = 0
        while frames_collected < frames_per_seq:
            ret, frame = cap.read()
            if not ret:
                print("[⚠️] Webcam frame read failed.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            landmarks = []
            if results.multi_hand_landmarks:
                print(f"[DEBUG] Hand(s) detected: {len(results.multi_hand_landmarks)}")
                for hand_idx, hand in enumerate(results.multi_hand_landmarks[:2]):  # Up to 2 hands
                    for lm in hand.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])  # 3D coordinates
                    # Pad if less than 42 landmarks for this hand
                    while len(landmarks) < (hand_idx + 1) * 126:
                        landmarks.extend([0.0, 0.0, 0.0])
                # Ensure 252 features (126 per hand)
                while len(landmarks) < 252:
                    landmarks.extend([0.0, 0.0, 0.0])
                landmarks = landmarks[:252]  # Cap at 252
                print(f"[DEBUG] Frame {frames_collected} landmarks (first 10): {landmarks[:10]}")
            else:
                print(f"[DEBUG] No hand detected in frame {frames_collected}")
                landmarks = [0.0] * 252  # Pad with zeros if no detection

            sequence.append(landmarks)
            frames_collected += 1

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Seq {seq_num+1}/{num_sequences} Frame {frames_collected}/{frames_per_seq}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Collecting Gesture", frame)

            if cv2.waitKey(1) == ord('q'):
                print("[🛑] Interrupted by user.")
                raise KeyboardInterrupt()

        if len(sequence) == frames_per_seq and all(len(seq) == 252 for seq in sequence):
            np.save(f"{save_dir}/{existing + seq_num}.npy", np.array(sequence))
            print(f"[✅] Saved sequence {existing + seq_num} for {gesture}")
            # Verify sample data from saved sequence
            sample_data = sequence[0]
            print(f"[DEBUG] Sample data for sequence {existing + seq_num} (first 10): {sample_data[:10]}")
        else:
            print(f"[❌] Skipping sequence {seq_num + 1} (inconsistent shape or not enough frames)")

    except Exception as e:
        print(f"[⚠️] Exception in sequence {seq_num + 1}: {e}")
        traceback.print_exc()

print("[🎉] Collection loop finished.")
cap.release()
cv2.destroyAllWindows()
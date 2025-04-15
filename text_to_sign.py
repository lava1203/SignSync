import cv2
import os

media_folder = "static/signs_gif"
text_input = input("Enter a word: ").strip().lower().replace(" ", "")
media_path = os.path.join(media_folder, f"{text_input}.mp4")

if os.path.exists(media_path):
    while True:
        cap = cv2.VideoCapture(media_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break inner loop when video ends to restart it
            cv2.imshow("Sign Output", frame)

            # Break both loops if 'q' is pressed
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        cap.release()
else:
    print(f"No sign found for '{text_input}'")

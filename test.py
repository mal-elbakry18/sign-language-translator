import cv2
import numpy as np
import mediapipe as mp
import requests

# Flask backend URL
API_URL = "http://127.0.0.1:5000/predict"

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def live_stream_and_send_to_backend():
    cap = cv2.VideoCapture(0)
    prediction = "..."
    sequence = []

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w, _ = frame_bgr.shape
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # Use webcam frame as background
            annotated_frame = frame_bgr.copy()

            # Draw landmarks (hands + pose) on webcam background
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Resize and normalize
            skeleton_resized = cv2.resize(annotated_frame, (224, 224))
            skeleton_normalized = skeleton_resized.astype("float32") / 255.0
            sequence.append(skeleton_normalized)

            # When 30 frames collected, send to Flask backend
            if len(sequence) == 30:
                try:
                    payload = {
                        "frames": np.expand_dims(sequence, axis=0).tolist()  # Shape: (1, 30, 224, 224, 3)
                    }
                    response = requests.post(API_URL, json=payload)
                    result = response.json()
                    prediction = result.get("word") or result.get("message", "...")
                except Exception as e:
                    prediction = f"Error: {str(e)}"
                sequence.clear()

            # Display live prediction
            cv2.putText(annotated_frame, f"Prediction: {prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Sign Recognition (Flask Backend)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_stream_and_send_to_backend()

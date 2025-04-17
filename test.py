# Install required libraries
#!pip install mediapipe opencv-python

# Import necessary libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands solution
hands = mp_hands.Hands(
    static_image_mode=False,        # For real-time video, keep this False
    max_num_hands=2,                # Detect up to 2 hands
    min_detection_confidence=0.5,   # Minimum confidence for hand detection
    min_tracking_confidence=0.5     # Minimum confidence for tracking
)

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# Function to recognize gestures
def recognize_gesture(hand_landmarks):
    # Extract key landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Example Gesture: "Thumbs Up"
    thumb_distance = calculate_distance(thumb_tip, wrist)
    index_distance = calculate_distance(index_tip, wrist)

    if thumb_distance > index_distance * 1.5:
        return "Thumbs Up"

    # Example Gesture: "Open Palm" (all fingers extended)
    if all(hand_landmarks.landmark[f].y < wrist.y for f in [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]):
        return "Open Palm"

    return "Unknown Gesture"

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open the webcam.")
else:
    print("Webcam opened. Press 'q' to quit.")

# Real-time video loop
try:
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read a frame.")
            break

        # Convert the frame to RGB (MediaPipe requires RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(frame_rgb)

        # Draw hand landmarks and recognize gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Recognize gesture
                gesture = recognize_gesture(hand_landmarks)
                cv2.putText(
                    frame, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                )

        # Display the frame with landmarks and gesture
        cv2.imshow('Real-Time Gesture Recognition', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {str(e)}")

# Release resources
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('improved_hand_gesture_model.h5')

# Define the MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Define the class labels
class_labels = np.load('y.npy')
unique_labels = np.unique(class_labels)
label_dict = {i: label for i, label in enumerate(unique_labels)}

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # optional mirror
    h, w, _ = frame.shape

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe hands
    results = hands.process(rgb_frame)

    # >>> Blur the whole frame first
    blurred = cv2.GaussianBlur(frame, (35, 35), 30)
    output_frame = blurred.copy()

    if results.multi_hand_landmarks:
        landmarks = []
        xs, ys = [], []

        for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks_list = []
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                xs.append(x)
                ys.append(y)
                hand_landmarks_list.append(landmark.x)
                hand_landmarks_list.append(landmark.y)
            landmarks.append(hand_landmarks_list)

        if len(landmarks) == 2:
            # Flatten the landmarks list
            landmarks = [item for sublist in landmarks for item in sublist]

            # Make predictions using the model
            predictions = model.predict(np.array([landmarks]), verbose=0)
            predicted_class = np.argmax(predictions)
            predicted_label = label_dict[predicted_class]

            # Bounding box around both hands
            x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
            y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)

            # Replace blurred region with original (hands stay sharp)
            output_frame[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]

            # Draw the predicted label
            cv2.putText(output_frame, predicted_label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    # Display the processed frame
    cv2.imshow('Hand Gesture Recognition', output_frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

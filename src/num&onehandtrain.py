import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# Define constants
NUM_HAND_LANDMARKS = 21
NUM_CLASSES = len(os.listdir('numerical_landmarks'))

# Load hand landmark data
def load_data(directory):
    data = []
    labels = []
    label_mapping = {}
    label_index = 0
    for subfolder in os.listdir(directory):
        if subfolder not in label_mapping:
            label_mapping[subfolder] = label_index
            label_index += 1
        subfolder_path = os.path.join(directory, subfolder)
        for file in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()[1:]  # skip the first line
                landmarks = []
                for line in lines:
                    parts = line.split(',')
                    x_part = [part for part in parts if 'x=' in part]
                    y_part = [part for part in parts if 'y=' in part]
                if x_part and y_part:
                    x = float(parts[0].split('=')[1])
                    y = float(parts[1].split('=')[1])
                    landmarks.append(x)
                    landmarks.append(y)
                    landmarks += [0] * (max_landmarks - len(landmarks))  # pad with zeros
                    data.append(landmarks)
                labels.append(label_mapping[subfolder])

    return np.array(data), np.array(labels)


# Load data
data, labels = load_data('numerical_landmarks')

# Split data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(NUM_HAND_LANDMARKS * 2,)))
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(test_data, test_labels))
model.save('numericalhand_gesture_model2.0.h5')



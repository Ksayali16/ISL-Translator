import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Load the trained model
model = tf.keras.models.load_model('image_hand_gesture_model.h5')

# Load class mapping
class_indices = np.load('class_mapping.npy', allow_pickle=True).item()
# Invert class_indices for lookup {index: class_name}
class_mapping = {v: k for k, v in class_indices.items()}

def predict_image(img_path, target_size=(64, 64)):
    # Load and preprocess image
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Rescale like training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    predicted_class = class_mapping.get(predicted_class_index, "Unknown")
    return predicted_class, confidence

# Example usage
img_path = r"C:\Users\Admin\Documents\Btechproject\Indian\A\0.jpg"
predicted_class, confidence = predict_image(img_path)
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}")

# Predicting for multiple images in a folder
def predict_folder(folder_path, target_size=(64, 64)):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):  # Check if file
            try:
                predicted_class, confidence = predict_image(img_path, target_size)
                print(f"Image: {img_name}, Predicted: {predicted_class}, Confidence: {confidence:.2f}")
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

# Example folder prediction
folder_path = r"path_to_test_folder"
predict_folder(folder_path)



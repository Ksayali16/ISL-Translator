import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# =========================
# 1️⃣ Load data
# =========================
X = np.load('X.npy')  # shape (num_samples, num_landmarks*2)
y = np.load('y.npy')  # shape (num_samples,) with letters 'A', 'B', etc.

# =========================
# 2️⃣ Encode labels
# =========================
le = LabelEncoder()
y_int = le.fit_transform(y)  # letters → integers
num_classes = len(np.unique(y_int))
y_cat = to_categorical(y_int, num_classes=num_classes)

# Save label encoder for decoding predictions later
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# =========================
# 3️⃣ Normalize landmarks
# =========================
def normalize_landmarks(X):
    X_norm = []
    for sample in X:
        lm = sample.reshape(-1, 2)
        center = np.mean(lm, axis=0)
        lm -= center
        max_val = np.max(np.linalg.norm(lm, axis=1))
        if max_val > 0:
            lm /= max_val
        X_norm.append(lm.flatten())
    return np.array(X_norm)

X = normalize_landmarks(X)

# =========================
# 4️⃣ Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_int
)

# =========================
# 5️⃣ Compute class weights
# =========================
class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=y_int
)
class_weights = dict(enumerate(class_weights_values))

# =========================
# 6️⃣ Build deeper model
# =========================
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# =========================
# 7️⃣ Train model
# =========================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# =========================
# 8️⃣ Save model
# =========================
model.save('improved_hand_gesture_model.h5')
print("✅ Model trained and saved successfully!")

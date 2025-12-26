import kagglehub
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# =====================================
# DOWNLOAD DATASET SAFELY
# =====================================
path = kagglehub.dataset_download("gti-upm/leapgestrecog")
print("Downloaded to:", path)

# Find leapGestRecog folder dynamically
dataset_path = None
for root, dirs, files in os.walk(path):
    if "leapGestRecog" in dirs:
        dataset_path = os.path.join(root, "leapGestRecog")
        break

if dataset_path is None:
    raise FileNotFoundError("leapGestRecog folder not found")

print("Using dataset:", dataset_path)

# =====================================
# LOAD DATA
# =====================================
IMG_SIZE = 64
X, y = [], []
gesture_labels = {}
label_index = 0

for subject in sorted(os.listdir(dataset_path)):
    subject_path = os.path.join(dataset_path, subject)
    if not os.path.isdir(subject_path):
        continue

    for gesture in sorted(os.listdir(subject_path)):
        gesture_path = os.path.join(subject_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        if gesture not in gesture_labels:
            gesture_labels[gesture] = label_index
            label_index += 1

        label = gesture_labels[gesture]

        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
print("Gesture classes:", gesture_labels)

# =====================================
# PREPROCESS
# =====================================
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = to_categorical(y, num_classes=len(gesture_labels))

# =====================================
# TRAIN / TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# CNN MODEL
# =====================================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(gesture_labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# TRAIN
# =====================================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# =====================================
# EVALUATE
# =====================================
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# =====================================
# SAVE MODEL
# =====================================
model.save("hand_gesture_model.h5")
print("Model saved as hand_gesture_model.h5")

# =====================================
# SHOW SAMPLE PREDICTIONS
# =====================================
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    pred = np.argmax(model.predict(X_test[i:i+1]))
    label_name = list(gesture_labels.keys())[pred]
    plt.title(label_name)
    plt.axis("off")

plt.tight_layout()
plt.show()

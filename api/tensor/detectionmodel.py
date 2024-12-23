import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

# Set the path for your local data directories
train_dir = 'path/to/your/train_data'
test_dir = 'path/to/your/test_data'

# Prepare data generators for training and testing
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128), 
    batch_size=32,        
    class_mode='binary'     # Binary because we have (productive/unproductive)
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Load the MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,         # Exclude final classification layers, we want our own
    weights='imagenet'         # Use pretrained weights from ImageNet
)

# Freeze the base model
base_model.trainable = False

# Build the model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)

# Load and preprocess the image for prediction
image_path = 'path/to/your/uploaded_image.jpg'  # Replace with the actual image path

img = load_img(image_path, target_size=(128, 128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict the class of the uploaded image
prediction = model.predict(img_array)
print("Prediction:", "Productive" if prediction[0][0] < 0.5 else "Unproductive")

# Save the model
model.save('model.h5')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Directories for training and validation data
train_dir = 'splitdata/train'
validation_dir = 'splitdata/val'

# Data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load data from directories with target size 128x128
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Change target size to 128x128
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),  # Change target size to 128x128
    batch_size=32,
    class_mode='categorical'
)

# Convert DirectoryIterator to tf.data.Dataset and repeat the dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),  # Update shape to 128x128x3
        tf.TensorSpec(shape=(None, 35), dtype=tf.float32)
    )
).repeat()  # Repeat indefinitely

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),  # Update shape to 128x128x3
        tf.TensorSpec(shape=(None, 35), dtype=tf.float32)
    )
)

# Model definition with input size 128x128x3
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(35, activation='softmax')  # Assuming 35 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the trained model
model.save("sign_language_model.h5")
print("Model saved as 'sign_language_model.h5'")

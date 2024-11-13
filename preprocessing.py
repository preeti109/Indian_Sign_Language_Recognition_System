from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Ensure that the directories exist
print(os.path.exists('splitdata/train'))  # Should return True if the directory exists
print(os.path.exists('splitdata/val'))    # Should return True if the directory exists

# Use raw strings or double backslashes to define paths
train_dir = r"splitdata\train"
val_dir = r"splitdata\val"

# Image dimensions and batch size
img_height, img_width = 64, 64
batch_size = 32

# Data Augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,               # Normalize pixel values to [0, 1]
    rotation_range=20,             # Random rotations
    width_shift_range=0.2,         # Random horizontal shifts
    height_shift_range=0.2,        # Random vertical shifts
    shear_range=0.2,               # Random shear transformations
    zoom_range=0.2,                # Random zooms
    horizontal_flip=True           # Random horizontal flip
)

# For validation, only rescale the images (no augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Set up the training generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),  # Resize images
    batch_size=batch_size,
    class_mode='categorical'              # Categorical for multi-class classification
)

# Set up the validation generator
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),  # Resize images
    batch_size=batch_size,
    class_mode='categorical'              # Categorical for multi-class classification
)

# Print out the class indices (to check the labels)
print("Classes: ", train_generator.class_indices)

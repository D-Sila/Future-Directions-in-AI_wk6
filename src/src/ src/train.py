"""
train.py
Lightweight CNN model for recyclable item classification.
Includes data loading, augmentation, training, and model saving.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# CONFIGURATION
# ----------------------------
IMG_SIZE = (128, 128)       # Input image resolution
BATCH_SIZE = 32             # Number of images per batch
EPOCHS = 20                 # Training epochs
DATA_DIR = "/path/to/data"  # Folder with 'train/', 'val/', 'test/' directories

# ----------------------------
# DATA PREPARATION
# ----------------------------
# ImageDataGenerator handles augmentation + rescaling pixel values.
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.15   # Split data into train/validation subsets
)

# Training data loader
train_flow = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='training'
)

# Validation data loader
val_flow = train_gen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset='validation'
)

num_classes = train_flow.num_classes  # Number of categories detected from folders

# ----------------------------
# MODEL DEFINITION
# ----------------------------
def build_model(input_shape=IMG_SIZE + (3,), num_classes=num_classes):
    """
    Builds a small CNN suitable for Edge devices (Raspberry Pi, TFLite).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Convolution + ReLU activation
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),

        # GlobalAveragePooling2D reduces model size vs Flatten()
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),  # Helps reduce overfitting

        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

model = build_model()

# ----------------------------
# MODEL COMPILATION
# ----------------------------
model.compile(
    optimizer='adam',                     # Good general-purpose optimizer
    loss='categorical_crossentropy',      # For multi-class classification
    metrics=['accuracy']                  # Track accuracy during training
)

# ----------------------------
# TRAINING THE MODEL
# ----------------------------
history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS
)

# ----------------------------
# SAVE TRAINED MODEL
# ----------------------------
model.save("models/recycle_model.h5")     # Save for later TFLite conversion

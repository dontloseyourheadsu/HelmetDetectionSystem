import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- CONFIGURATION ---
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
EPOCHS = 15  # How many times to go through the dataset

# --- 1. SETUP PATHS ---
# Based on your screenshot, this script is in /cnn/
# and images are in /filtering/processed/
current_dir = pathlib.Path(__file__).parent.resolve()
data_dir = current_dir.parent / "filtering" / "processed"

# (Snippet for Approach 2 - Anomaly Detection)
# Only load the 'clean' folder for training
clean_dir = data_dir / "clean" 

train_ds = tf.keras.utils.image_dataset_from_directory(
    clean_dir, # pointing ONLY to clean
    label_mode=None, # We don't need labels, we just need the image itself
    color_mode='grayscale',
    image_size=(128, 128),
    batch_size=32
)

# In an autoencoder, the input is the image, and the target is ALSO the image.
# We train it to copy the input to the output.
train_ds = train_ds.map(lambda x: (x/255.0, x/255.0))
"""Convolutional Autoencoder for Anomaly Detection.

This script trains a Convolutional Autoencoder on clean images to learn the normal representation.
Anomalies (hair/trash) are detected when the reconstruction error exceeds a calculated threshold.
"""

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
EPOCHS = 20

# --- 1. SETUP PATHS & DATA LOADING ---
current_dir = pathlib.Path(__file__).parent.resolve()
data_dir = current_dir.parent / "filtering" / "processed"

clean_dir = data_dir / "clean"
hair_dir = data_dir / "hair"
trash_dir = data_dir / "trash"
trash_hair_dir = data_dir / "trash-hair"

def load_image(path: str) -> tf.Tensor:
    """Load and preprocess an image.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Normalized image tensor.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    return img

def get_dataset(file_paths: list) -> tf.data.Dataset:
    """Create a TensorFlow dataset from file paths.
    
    Args:
        file_paths: List of image file paths.
        
    Returns:
        TensorFlow dataset.
    """
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def get_files(directory: pathlib.Path, limit: int = None) -> list:
    """Get list of image files in a directory.
    
    Args:
        directory: Directory to search.
        limit: Maximum number of files to return.
        
    Returns:
        List of file paths.
    """
    if not directory.exists():
        return []
    files = sorted([str(p) for p in directory.glob("*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    if limit:
        return files[:limit]
    return files

# 1. Split Clean Data: First 100 for Test, Rest for Train
all_clean = get_files(clean_dir)
test_clean_paths = all_clean[:100]
train_clean_paths = all_clean[100:]

print(f"Clean Images: {len(train_clean_paths)} for Training, {len(test_clean_paths)} for Testing")

# 2. Get Anomaly Test Sets (First 100)
test_hair_paths = get_files(hair_dir, 100)
test_trash_paths = get_files(trash_dir, 100)
test_trash_hair_paths = get_files(trash_hair_dir, 100)

print(f"Anomaly Test Sets: Hair={len(test_hair_paths)}, Trash={len(test_trash_paths)}, Trash-Hair={len(test_trash_hair_paths)}")

# 3. Create Training Dataset
train_ds = get_dataset(train_clean_paths)
train_ds = train_ds.map(lambda x: (x, x)) 
train_ds = train_ds.batch(BATCH_SIZE).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

# --- 2. BUILD THE CONVOLUTIONAL AUTOENCODER ---
class ConvolutionalAutoencoder(models.Model):
  """Convolutional Autoencoder model."""
  
  def __init__(self):
    super(ConvolutionalAutoencoder, self).__init__()
    
    # Encoder: Compresses the image into spatial features
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
      layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2), # 64x64
      layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2), # 32x32
      layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2) # 16x16
    ])
    
    # Decoder: Reconstructs the image from features
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same'), # 32x32
      layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),  # 64x64
      layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),  # 128x128
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same') # Output layer
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def get_config(self):
    return super().get_config()

autoencoder = ConvolutionalAutoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

# --- 3. TRAIN ---
print("Starting training (Convolutional Autoencoder)...")
history = autoencoder.fit(
    train_ds,
    epochs=EPOCHS,
    shuffle=True
)

# --- 4. CALCULATE THRESHOLD ---
print("Calculating anomaly threshold based on training data...")
train_loss = []
for batch in train_ds:
    imgs = batch[0]
    recon = autoencoder(imgs)
    loss = tf.reduce_mean(tf.abs(imgs - recon), axis=(1, 2, 3))
    train_loss.extend(loss.numpy())

train_loss = np.array(train_loss)
# Adjusted Threshold: Mean + 1.5 * STD (Tighter than 2*STD to catch more anomalies)
threshold = np.mean(train_loss) + 1.5 * np.std(train_loss) 

print(f"Mean Loss: {np.mean(train_loss):.4f}, Std Dev: {np.std(train_loss):.4f}")
print(f"Anomaly Threshold: {threshold:.4f} (Mean + 1.5*STD)")

# --- 5. EVALUATE PERFORMANCE ---
def evaluate_set(name: str, file_paths: list, expected_is_anomaly: bool) -> np.ndarray:
    """Evaluate the model on a dataset.
    
    Args:
        name: Name of the dataset.
        file_paths: List of image file paths.
        expected_is_anomaly: Whether the dataset contains anomalies.
        
    Returns:
        Array of loss values.
    """
    if not file_paths:
        print(f"Skipping {name} (no files)")
        return
    
    ds = get_dataset(file_paths).batch(BATCH_SIZE)
    losses = []
    for imgs in ds:
        recon = autoencoder(imgs)
        loss = tf.reduce_mean(tf.abs(imgs - recon), axis=(1, 2, 3))
        losses.extend(loss.numpy())
    
    losses = np.array(losses)
    predictions = (losses > threshold).astype(int) # 1 = Anomaly, 0 = Clean
    
    # Metrics
    total = len(predictions)
    anomalies_detected = np.sum(predictions)
    
    if expected_is_anomaly:
        accuracy = anomalies_detected / total
        print(f"[{name}] Accuracy (Recall): {accuracy:.2%} ({anomalies_detected}/{total} detected as anomaly)")
    else:
        accuracy = (total - anomalies_detected) / total
        print(f"[{name}] Accuracy (Specificity): {accuracy:.2%} ({total - anomalies_detected}/{total} detected as clean)")

    return losses

print("\n--- Performance Evaluation ---")
evaluate_set("Test Clean", test_clean_paths, expected_is_anomaly=False)
evaluate_set("Test Hair", test_hair_paths, expected_is_anomaly=True)
evaluate_set("Test Trash", test_trash_paths, expected_is_anomaly=True)
evaluate_set("Test Trash-Hair", test_trash_hair_paths, expected_is_anomaly=True)

# --- 6. VISUALIZE RESULTS (CLEAN vs ANOMALY) ---
print("\nVisualizing reconstruction...")

def get_one_batch(paths: list) -> tf.Tensor:
    """Get a single batch of images."""
    if not paths: return None
    return next(iter(get_dataset(paths).batch(10)))

x_clean = get_one_batch(test_clean_paths)
x_anom = get_one_batch(test_hair_paths)

if x_clean is not None:
    encoded_clean = autoencoder.encoder(x_clean).numpy()
    decoded_clean = autoencoder.decoder(encoded_clean).numpy()

if x_anom is not None:
    encoded_anom = autoencoder.encoder(x_anom).numpy()
    decoded_anom = autoencoder.decoder(encoded_anom).numpy()

# Plot Comparison
n = 5
plt.figure(figsize=(15, 6))
plt.suptitle(f"Convolutional Autoencoder Evaluation (Threshold: {threshold:.4f})", fontsize=16)

if x_clean is not None:
    # Row 1: Original Clean
    for i in range(n):
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(x_clean[i].numpy().reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title("Clean (Orig)")
        plt.axis("off")

    # Row 2: Reconstructed Clean
    for i in range(n):
        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(decoded_clean[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        loss = np.mean(np.abs(x_clean[i] - decoded_clean[i]))
        color = 'green' if loss < threshold else 'red'
        plt.title(f"Loss: {loss:.4f}", color=color)
        plt.axis("off")

if x_anom is not None:
    # Row 3: Original Anomaly
    for i in range(n):
        ax = plt.subplot(4, n, i + 1 + 2*n)
        plt.imshow(x_anom[i].numpy().reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        plt.title("Anomaly (Orig)")
        plt.axis("off")

    # Row 4: Reconstructed Anomaly
    for i in range(n):
        ax = plt.subplot(4, n, i + 1 + 3*n)
        plt.imshow(decoded_anom[i].reshape(IMG_HEIGHT, IMG_WIDTH), cmap='gray')
        loss = np.mean(np.abs(x_anom[i] - decoded_anom[i]))
        color = 'green' if loss > threshold else 'red'
        plt.title(f"Loss: {loss:.4f}", color=color)
        plt.axis("off")

plt.tight_layout()
plt.savefig('results/autoencoder_reconstruction.png')
plt.show()

# --- 7. SAVE MODEL ---
autoencoder.save('clean_field_autoencoder_cnn.keras')
print("Model saved as clean_field_autoencoder_cnn.keras")

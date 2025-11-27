import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import threading
import queue
import time
import os

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128

def remove_large_noise(tophat_image: np.ndarray, threshold_value: int = 15, max_area: int = 80) -> np.ndarray:
    """Filters out large bright objects (like logos) from the tophat layer."""
    _, binary = cv2.threshold(tophat_image, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(tophat_image)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0 < area < max_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
    result = cv2.bitwise_and(tophat_image, tophat_image, mask=clean_mask)
    return result

def morphological_contrast_enhancement(image: np.ndarray, kernel_size: int = 19, crumb_boost: float = 4.0, hair_boost: float = 4.0, shadow_gamma: float = 0.6) -> np.ndarray:
    """Enhances hair (dark) and crumbs (light) by extracting them and placing them on a neutral gray background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    norm_img = smoothed.astype(np.float32) / 255.0
    lifted = np.power(norm_img, shadow_gamma)
    lifted_uint8 = (lifted * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    white_tophat = cv2.morphologyEx(lifted_uint8, cv2.MORPH_TOPHAT, kernel)
    white_tophat_clean = remove_large_noise(white_tophat, threshold_value=10, max_area=60)
    black_tophat = cv2.morphologyEx(lifted_uint8, cv2.MORPH_BLACKHAT, kernel)
    flat_background = np.full_like(lifted_uint8, 128, dtype=np.float32)
    result = flat_background + (white_tophat_clean.astype(np.float32) * crumb_boost)
    result -= (black_tophat.astype(np.float32) * hair_boost)
    result = np.clip(result, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    final = clahe.apply(result)
    return final

@tf.keras.utils.register_keras_serializable()
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

class AutoencoderApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Load Model
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "..", "clean_field_autoencoder_cnn.keras")
            
            print(f"Loading model from: {model_path}")
            # Instantiate and load weights directly to avoid deserialization issues with subclassed models
            self.model = ConvolutionalAutoencoder()
            # Run a dummy input to build the model graph
            dummy_input = tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 1))
            self.model(dummy_input)
            # Load weights
            self.model.load_weights(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.window.destroy()
            return

        # Camera Setup
        self.video_source = 0
        self.vid = None
        self.is_monitoring = False
        self.is_calibrating = False
        
        # Anomaly Detection Settings
        self.threshold = 0.05 # Default starting value
        self.calibration_losses = []
        
        # UI Elements
        self.top_frame = tk.Frame(window)
        self.top_frame.pack(pady=10)
        
        self.lbl_camera = tk.Label(self.top_frame, text="Camera Index:")
        self.lbl_camera.pack(side=tk.LEFT, padx=5)
        
        self.camera_entry = tk.Entry(self.top_frame, width=5)
        self.camera_entry.insert(0, "0")
        self.camera_entry.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(self.top_frame, text="Start Monitoring", command=self.toggle_monitoring, bg="green", fg="white", width=15)
        self.btn_start.pack(side=tk.LEFT, padx=10)

        self.btn_calibrate = tk.Button(self.top_frame, text="Calibrate (5s)", command=self.start_calibration, bg="orange", fg="black", width=15)
        self.btn_calibrate.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(window, width=640, height=480, bg="black")
        self.canvas.pack()
        
        # Controls Frame
        self.controls_frame = tk.Frame(window)
        self.controls_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.lbl_threshold = tk.Label(self.controls_frame, text=f"Threshold: {self.threshold:.4f}")
        self.lbl_threshold.pack(side=tk.LEFT)
        
        self.scale_threshold = tk.Scale(self.controls_frame, from_=0.0, to=0.2, resolution=0.001, orient=tk.HORIZONTAL, length=400, command=self.update_threshold_from_scale)
        self.scale_threshold.set(self.threshold)
        self.scale_threshold.pack(side=tk.LEFT, padx=10)

        self.lbl_status = tk.Label(window, text="Status: Idle", font=("Helvetica", 14))
        self.lbl_status.pack(pady=5)
        
        self.lbl_loss = tk.Label(window, text="Current Loss: 0.0000", font=("Helvetica", 12))
        self.lbl_loss.pack(pady=5)

        # Processing Setup
        self.frame_queue = queue.Queue()
        self.last_sample_time = 0
        self.sample_interval = 2.0 # seconds
        self.stop_event = threading.Event()
        
        self.processing_thread = threading.Thread(target=self.process_queue, daemon=True)
        self.processing_thread.start()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.delay = 15 # ms
        self.update()

    def update_threshold_from_scale(self, val):
        self.threshold = float(val)
        self.lbl_threshold.config(text=f"Threshold: {self.threshold:.4f}")

    def toggle_monitoring(self):
        if self.is_monitoring:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        try:
            src_str = self.camera_entry.get()
            if not src_str.isdigit():
                messagebox.showerror("Error", "Camera index must be a number.")
                return
                
            src = int(src_str)
            self.vid = cv2.VideoCapture(src)
            if not self.vid.isOpened():
                raise ValueError(f"Unable to open camera {src}")
            
            self.is_monitoring = True
            self.btn_start.config(text="Stop Monitoring", bg="red")
            self.lbl_status.config(text="Status: Monitoring...", fg="blue")
            self.last_sample_time = time.time()
            
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            if self.vid:
                self.vid.release()
                self.vid = None

    def stop_monitoring(self):
        self.is_monitoring = False
        self.is_calibrating = False
        if self.vid:
            self.vid.release()
            self.vid = None
        self.btn_start.config(text="Start Monitoring", bg="green")
        self.lbl_status.config(text="Status: Idle", fg="black")
        self.canvas.delete("all")
        
        # Clear queue
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

    def start_calibration(self):
        if not self.is_monitoring:
            self.start_monitoring()
        
        if self.is_monitoring:
            self.is_calibrating = True
            self.calibration_losses = []
            self.lbl_status.config(text="Status: Calibrating... (Please ensure field is clean)", fg="orange")
            # Temporarily speed up sampling for calibration
            self.original_interval = self.sample_interval
            self.sample_interval = 0.5 
            
            # Stop calibration after 5 seconds (approx 10 samples)
            self.window.after(5000, self.finish_calibration)

    def finish_calibration(self):
        self.is_calibrating = False
        self.sample_interval = 2.0 # Restore interval
        
        if self.calibration_losses:
            losses = np.array(self.calibration_losses)
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            # Set threshold to Mean + 3 * STD (Standard statistical outlier detection)
            new_threshold = mean_loss + 3 * std_loss
            
            self.threshold = new_threshold
            self.scale_threshold.set(new_threshold)
            self.lbl_threshold.config(text=f"Threshold: {self.threshold:.4f}")
            
            messagebox.showinfo("Calibration Complete", f"New Threshold set to: {self.threshold:.4f}\n(Mean: {mean_loss:.4f}, Std: {std_loss:.4f})")
            self.lbl_status.config(text="Status: Monitoring...", fg="blue")
        else:
            messagebox.showwarning("Calibration Failed", "No frames captured during calibration.")
            self.lbl_status.config(text="Status: Monitoring...", fg="blue")

    def update(self):
        if self.is_monitoring and self.vid:
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                current_time = time.time()
                if current_time - self.last_sample_time >= self.sample_interval:
                    self.frame_queue.put(frame.copy())
                    self.last_sample_time = current_time
            else:
                self.stop_monitoring()
                messagebox.showerror("Error", "Lost connection to camera.")
        
        self.window.after(self.delay, self.update)

    def process_queue(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if not self.is_monitoring:
                continue

            try:
                # Preprocess with filters
                filtered_frame = morphological_contrast_enhancement(
                    frame, 
                    kernel_size=19,
                    crumb_boost=5.0, 
                    hair_boost=5.0,    
                    shadow_gamma=0.6   
                )
                
                # Resize to model input size
                resized = cv2.resize(filtered_frame, (IMG_WIDTH, IMG_HEIGHT))
                normalized = resized.astype('float32') / 255.0
                
                # Add batch dimension: (1, 128, 128)
                input_data = np.expand_dims(normalized, axis=0)
                # Add channel dimension: (1, 128, 128, 1)
                input_data = np.expand_dims(input_data, axis=-1)
                
                # Predict (Reconstruct)
                reconstruction = self.model.predict(input_data, verbose=0)
                
                # Calculate Loss (MAE - Mean Absolute Error)
                # Matches training logic: tf.reduce_mean(tf.abs(imgs - recon))
                loss = np.mean(np.abs(input_data - reconstruction))
                
                # Update UI
                self.window.after(0, lambda l=loss: self.update_loss_label(l))
                
                if self.is_calibrating:
                    self.calibration_losses.append(loss)
                else:
                    if loss > self.threshold:
                        self.window.after(0, lambda l=loss: self.handle_detection(l))
                        
            except Exception as e:
                print(f"Error in processing thread: {e}")

    def update_loss_label(self, loss):
        color = "red" if loss > self.threshold else "green"
        self.lbl_loss.config(text=f"Current Loss: {loss:.4f}", fg=color)

    def handle_detection(self, loss):
        if not self.is_monitoring or self.is_calibrating:
            return
            
        self.stop_monitoring()
        messagebox.showwarning("Anomaly Detected!", f"Reconstruction Error: {loss:.4f}\n(Threshold: {self.threshold:.4f})\n\nPotential Contaminant Detected!")
        self.start_monitoring()

    def on_closing(self):
        self.stop_event.set()
        self.stop_monitoring()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AutoencoderApp(root, "Sterile Field Autoencoder Detector")
    root.mainloop()

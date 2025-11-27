import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import PIL.Image, PIL.ImageTk
import tensorflow as tf
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
# Class names based on the folder structure in filtering/processed/
# tf.keras.utils.image_dataset_from_directory sorts them alphanumerically
CLASS_NAMES = ['clean', 'hair', 'trash', 'trash-hair'] 
CONTAMINANTS = ['hair', 'trash', 'trash-hair']

class ClassifierApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Load Model
        try:
            # Assuming the script is run from the root or apps/ folder, we look for the model in the root
            # Adjust path logic to find the model relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "..", "contamination_classifier.keras")
            
            print(f"Loading model from: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.window.destroy()
            return

        # Camera Setup
        self.video_source = 0
        self.vid = None
        self.is_monitoring = False
        
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

        self.canvas = tk.Canvas(window, width=640, height=480, bg="black")
        self.canvas.pack()
        
        self.lbl_status = tk.Label(window, text="Status: Idle", font=("Helvetica", 14))
        self.lbl_status.pack(pady=10)
        
        self.lbl_prediction = tk.Label(window, text="Last Prediction: None", font=("Helvetica", 10))
        self.lbl_prediction.pack(pady=5)

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
        if self.vid:
            self.vid.release()
            self.vid = None
        self.btn_start.config(text="Start Monitoring", bg="green")
        self.lbl_status.config(text="Status: Idle", fg="black")
        self.canvas.delete("all")
        
        # Clear queue
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

    def update(self):
        if self.is_monitoring and self.vid:
            ret, frame = self.vid.read()
            if ret:
                # Resize for display if needed, but 640x480 is standard
                frame = cv2.resize(frame, (640, 480))
                
                # Display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_rgb))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                # Sampling
                current_time = time.time()
                if current_time - self.last_sample_time >= self.sample_interval:
                    # Copy frame for processing
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
                
                # Add batch dimension: (1, 128, 128)
                input_data = np.expand_dims(resized, axis=0)
                # Add channel dimension: (1, 128, 128, 1)
                input_data = np.expand_dims(input_data, axis=-1)
                
                # Predict
                predictions = self.model.predict(input_data, verbose=0)
                score = tf.nn.softmax(predictions[0])
                class_idx = np.argmax(score)
                class_name = CLASS_NAMES[class_idx]
                confidence = 100 * np.max(score)
                
                print(f"Prediction: {class_name} ({confidence:.2f}%)")
                
                # Update UI with prediction (thread-safe way)
                self.window.after(0, lambda cn=class_name, conf=confidence: self.update_prediction_label(cn, conf))
                
                if class_name in CONTAMINANTS:
                    self.window.after(0, lambda cn=class_name: self.handle_detection(cn))
            except Exception as e:
                print(f"Error in processing thread: {e}")

    def update_prediction_label(self, class_name, confidence):
        self.lbl_prediction.config(text=f"Last Prediction: {class_name} ({confidence:.1f}%)")

    def handle_detection(self, class_name):
        # Double check monitoring state to avoid multiple popups
        if not self.is_monitoring:
            return
            
        self.stop_monitoring()
        messagebox.showwarning("Contaminant Detected!", f"Detected: {class_name}\nRecording stopped.")
        # When user closes the popup, we resume
        self.start_monitoring()

    def on_closing(self):
        self.stop_event.set()
        self.stop_monitoring()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ClassifierApp(root, "Sterile Field Contaminant Detector")
    root.mainloop()

# Destructible Vinyl ("Eggshell") Sticker Tampering Detection

Project goal
- Build a lightweight, easy-to-train system (suitable for Google Colab free tier) that classifies destructible vinyl security stickers as either:
  - `Intact` (no tampering), or
  - `Tampered` (attempted removal which causes the sticker to shatter/fragment).
- Approach: Pre-process images with Canny edge detection to extract fracture/edge maps, then train a small custom CNN on these edge maps.

Why this problem and approach
- Destructible vinyl stickers break into many small fragments when tampered with; the resulting fracture lines produce high-frequency edge patterns.
- Applying a Canny filter before training simplifies the input representation (black/white edge maps), letting a small CNN learn pattern differences faster and with less data and compute than training on RGB images.
- This is beginner-friendly, trains quickly on Colab, and focuses the model on the critical signal (fracture lines).

Quick overview / focused plan
1. Objective
   - Binary classification (Intact vs Tampered) using Canny edge maps → small CNN.

2. Dataset (DIY)
   - Acquire destructible vinyl stickers (commercial eggshell/destructible labels).
   - Apply stickers to a variety of surfaces (plastic, cardboard, metal).
   - Capture `Intact` photos: multiple images per sticker under varied lighting and slight angle changes.
   - Induce tampering: peel, scrape, or cut to cause fracturing; capture `Tampered` photos.
   - Organize images into:
     - data/raw/intact/
     - data/raw/tampered/
   - Target initially: 100–200 images per class. (More is better, but Colab-friendly training can work with a few hundred images when using Canny.)

3. Image preprocessing (OpenCV)
   - Pipeline per image:
     1. Load image
     2. Convert to grayscale
     3. Optional Gaussian blur (e.g., 5x5) to reduce sensor noise but preserve edges
     4. Apply Canny (tune lower/upper thresholds empirically)
     5. Save edge map (same folder structure under data/edges/)
   - Save edge maps (PNG) and use those as input to the CNN. This reduces variations due to color, texture, or lighting.

4. Model (beginner-friendly)
   - Framework: TensorFlow + Keras
   - Example architecture (small, custom):
     - Input: edge map (e.g., 128x128 or 224x224, single channel)
     - Conv2D(32, 3x3) → ReLU → MaxPool
     - Conv2D(64, 3x3) → ReLU → MaxPool
     - Conv2D(128, 3x3) → ReLU → MaxPool (optional)
     - Flatten → Dense(64) → ReLU → Dropout(0.5)
     - Output Dense(1) → Sigmoid
   - Loss: Binary crossentropy
   - Optimizer: Adam
   - Metrics: Accuracy, Precision, Recall, F1 (compute F1 in evaluation script)

5. Training & Platform
   - Use Google Colab (free tier). Typical flow:
     - Mount Google Drive for dataset persistence, or upload dataset to Colab session storage.
     - pip install -r requirements.txt (OpenCV, tensorflow, numpy, scikit-learn)
     - Run preprocessing script to create Canny edge maps.
     - Train with ImageDataGenerator or tf.data (use on-the-fly augmentation for edges: small rotations, shifts).
     - Keep image size small (128x128) to reduce memory/compute.

6. Evaluation & priorities
   - Use a held-out test set.
   - Report: Accuracy, Precision, Recall, F1-score.
   - Prioritize Recall for the `Tampered` class (we prefer false positives over missed tampering).
   - Compare Canny+CNN vs same CNN trained on raw grayscale images (control experiment) to demonstrate the filter's benefit.

Repository structure (recommended)
- README.md (this file)
- LICENSE
- data/
  - raw/
    - intact/
    - tampered/
  - edges/
    - train/
      - intact/
      - tampered/
    - val/
      - intact/
      - tampered/
- notebooks/
  - colab-experiments.ipynb
- scripts/
  - preprocess_canny.py        # script to convert raw images → edge maps
  - train_model.py             # Keras training script for edge maps
  - evaluate.py                # evaluation metrics & confusion matrix
- models/
  - checkpoints/
  - final/
- legacy/
  - helmet_detection/          # (optional) previous project artifacts
- .gitignore
- requirements.txt

Example preprocessing snippet (scripts/preprocess_canny.py)
```python
# This snippet is illustrative; full script should handle folders, logging, params
import cv2
import os

def make_edge_map(in_path, out_path, blur_ksize=(5,5), canny_thresh1=50, canny_thresh2=150):
    img = cv2.imread(in_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    cv2.imwrite(out_path, edges)
```

Example small Keras model (scripts/train_model.py — excerpt)
```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_small_cnn(input_shape=(128,128,1)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

Colab quickstart (notes)
1. Upload or mount dataset to /content/data/
2. Run preprocessing: python scripts/preprocess_canny.py --input data/raw --output data/edges --size 128
3. Train: python scripts/train_model.py --data data/edges --epochs 25 --batch-size 32
4. Evaluate: python scripts/evaluate.py --model models/final/best.h5 --data data/edges/test

Tips and hyperparameters to try
- Canny thresholds: start with (50, 150) and sweep (30-100, 100-200). Lighting and camera noise affect good thresholds.
- Image size: 128x128 is a good balance for Colab free tier.
- Augmentations: small rotations (±10°), small translations, flips (if meaningful).
- EarlyStopping on validation loss; save best checkpoint.

Ethics, safety, and limitations
- This system detects visual signs of physical tampering on a specific sticker type. Results are only as good as your dataset and capture conditions.
- False positives/negatives have consequences; design downstream processes (human-in-the-loop verification) accordingly.
- The approach is tailored to destructible vinyl stickers that create fracture edge patterns; it is not a general-purpose tamper-detection for every sticker type.

What changed from the previous project
- The original helmet detection focus is deprecated for this repository. This README documents the new task, repo structure suggestions, and scripts to build a Canny+CNN tamper detector. Move any reusable data loading, model utilities, or training scripts from the legacy helmet code into `scripts/` and adapt to single-channel (edge) inputs.

References and further reading (brief)
- Use OpenCV docs for Canny and image processing basics.
- TensorFlow/Keras tutorials for building and training CNNs.
- Research on crack and fracture detection (useful for feature inspiration).

Contributing
- If you want to help: contribute images (see data/README.md guidelines), improve preprocessing and augmentation strategies, or add more robust evaluation and CI.
- Open issues or PRs against this repository. Tag early-stage issues as `help-wanted` and `good-first-issue`.

Contact / Author
- Repository owner: dontloseyourheadsu
- If you want to fork or reuse, please credit the original repository and author.

License
- Add an appropriate license at the repository root (e.g., MIT) if you want the project to be reusable.
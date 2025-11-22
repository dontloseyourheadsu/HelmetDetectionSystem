Below is your README rewritten **in English**, keeping **all technical content identical**, only translating and formatting it professionally.

---

# Sterile-Field Micro-Contaminant Detector (SF-MCD)

**Project Status:** Proof-of-Concept

This project proposes a computer vision solution for detecting micro-contaminants (hair, debris, particles) on sterile surgical drapes. It uses a hybrid system composed of an advanced filtering pipeline and a lightweight CNN capable of operating with a standard webcam.

---

## 1. Core Problem: Low-Signal Target in a High-Noise Environment

Detecting contaminants on a sterile field involves two critical challenges:

**Low-Signal (Target):**

- The contaminant (hair, crumb, particle) is extremely small and low contrast.

**High-Noise (Background):**

- Shadows caused by folds in the surgical drape.
- Ambient lighting variations.
- High-contrast printed logos.
- Texture and chromatic variation of the material.

A conventional approach (grayscale preprocessing or a standard CNN) tends to confuse shadows and printed markings with contaminants, producing unacceptable false-positive rates.
The solution requires removing the noise before presenting the image to the model.

---

## 2. Overview of the “Filter-First” Pipeline

The system does not feed the original image to the CNN. Instead, it constructs a three-channel feature map produced through a five-stage pipeline:

1. Conversion to HSV
2. Logical masks (“mask algebra”)
3. Shape analysis for heuristic classification
4. Construction of the multichannel map
5. Training a simple CNN on the generated map

---

## 3. Pipeline Stages

### 3.1 HSV Conversion

The captured BGR image is converted to HSV to separate the color component (Hue) from the brightness component (Value).
This separation allows the system to distinguish drape color from shadows, glare, and white printed logos.

---

### 3.2 Mask Algebra

Binary masks are generated using `cv2.inRange()`:

- `mask_cloth`: identifies pixels matching the characteristic hue of the drape (blue, green, or pink).
- `mask_logo`: identifies white pixels (low saturation, high value).
- `mask_shadows_and_hair`: identifies dark regions (low value).

To remove the logo from the cloth mask:

```
mask_cloth_clean = cv2.subtract(mask_cloth, mask_logo)
```

---

### 3.3 Shape Analysis

The `mask_shadows_and_hair` mask contains both contaminants and shadows.
Geometric criteria are applied using `cv2.findContours()`:

- **Hair:** An ellipse is fitted using `cv2.fitEllipse`; if eccentricity > 4.0, the blob is classified as hair.
- **Trash:** Solidity (`area / convexHull`) is computed; if solidity > 0.85, the blob is classified as trash.
- **Shadow:** Any region that is large, low-solidity, or not elongated is discarded as shadow.

---

### 3.4 Construction of the Multichannel Map

Three clean masks are generated and merged using `cv2.merge()`:

- Blue Channel: drape area (`mask_cloth_clean`)
- Green Channel: trash (`mask_trash_final`)
- Red Channel: hair (`mask_hair_final`)

The CNN receives this structured three-channel feature map instead of the original image.

---

### 3.5 CNN Training

The model is trained exclusively on the multichannel maps.
Its task is reduced to detecting the presence of red or green regions over a blue background, which is far more stable and robust than learning directly from the original image.

---

## 4. Project Structure

```
Sterile-Field-Micro-Contaminant-Detector
│
├── src/
│   ├── data_collection/
│   │   ├── capture_video.py          # 1. Records .mp4 videos
│   │   └── extract_frames.py         # 2. Extracts frames from videos
│   │
│   ├── filtering/
│   │   ├── build_dataset.py          # 3. Advanced HSV + shape pipeline
│   │   └── processed_multichannel/   #    Output: multichannel maps
│   │
│   ├── training/
│   │   ├── model.py                  # 4. Simple CNN architecture
│   │   ├── train.py                  # 5. Model training
│   │   ├── sterile_field_model.keras #    Final trained model
│   │   └── training_history.png      #    Training performance plot
│
├── README.md
└── .gitignore
```

---

## 5. Usage Instructions

### 5.1 Data Collection

Run to record videos:

```
python src/data_collection/capture_video.py
```

Then extract frames:

```
python src/data_collection/extract_frames.py
```

---

### 5.2 Preprocessing (Filtering Pipeline)

Generate the multichannel maps:

```
python src/filtering/build_dataset.py
```

Output is stored in:

```
src/filtering/processed_multichannel/
```

---

### 5.3 CNN Training

```
python src/training/train.py
```

The resulting model is saved as:

```
src/training/sterile_field_model.keras
```

### 5.4 Binary Trash vs Safe Classification (Current Branch)

This branch adds a lightweight binary classification workflow to label crops as either `Trash` or `Safe` using a simple CNN.

1. Build the binary dataset from multi-class folders produced by boxing:

```
python dataset/prepare_binary_dataset.py --source crops
```

Options:

- Use `--source boxed` to pull from `boxing/boxed` instead of `boxing/crops`.
- Add `--link` to create symlinks instead of copying (saves disk space).
- Override mapping (defaults: Trash = `trash` + `trash-hair`, Safe = `clean` + `hair`):

```
python dataset/prepare_binary_dataset.py --trash trash trash-hair --safe clean hair
```

2. Train the binary CNN:

```
python cnn/cnn-classifier.py
```

This writes `trash_classifier_model.h5` and displays training curves. The console prints the folder-to-label mapping (`{'Safe': 0, 'Trash': 1}` expected) so you can confirm correct labeling.

Dataset folder produced:

```
CNN_Training_Data/
	Trash/
	Safe/
```

Ensure `tensorflow`, `matplotlib`, and `pillow` are installed (they were added to `requirements.txt`). Install/update dependencies:

```
pip install -r requirements.txt
```

You can regenerate the dataset any time if new crops are added.

---

### 5.4 Real-Time Inference (Future Work)

A future `inference.py` script will integrate:

- Webcam capture
- HSV + shape-filtering pipeline
- Inference using the trained model

---

## 6. Current Project Status

| Component                      | Status   |
| ------------------------------ | -------- |
| Data capture and extraction    | Complete |
| HSV pipeline + shape filtering | Complete |
| Multichannel map generation    | Complete |
| CNN training                   | Complete |
| Real-time inference            | Pending  |

---

If you want, I can also generate:

- A fully polished version suitable for publication
- A documentation-style version for medical device approval workflows
- A shorter or extended version for GitHub

Just tell me what tone or format you prefer.

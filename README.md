# Sterile-Field Micro-Contaminant Detector (SF-MCD)

**Project Status:** Proof-of-Concept

This project is a novel computer vision solution designed to detect micro-contaminants on sterile surgical fields using a hybrid "Filter + CNN" architecture.

The primary goal is to create a system that can run on a simple webcam and alert a surgical team if a sterile drape is compromised by a foreign object, such as a single hair, a fiber, or other small particles.

## The Core Problem: High-Noise, Low-Signal

In a medical environment, sterile fields are critical for preventing Surgical Site Infections (SSIs). A single contaminant, like a hair, can compromise the field.

This is a difficult computer vision problem for two reasons:

1.  **Low-Signal Target:** The contaminant (a single hair or a tiny scrap) is extremely small and has very low contrast against the background.
2.  **High-Noise Background:** The "sterile" drape itself is not a uniform surface. It is covered in wrinkles, folds, and shadows from operating room lights.

A simple Convolutional Neural Network (CNN) or a standard edge detector (like Sobel or Canny) will fail. The network will incorrectly learn that the "wrinkles" and "shadows"—which are much stronger and more frequent signals—are the features to detect, leading to an unusable rate of false positives.

**The solution is not to *find the hair*, but to *suppress the background*.**

## How It Works: The Filter-First Architecture

This project solves the "noise" problem by *never* showing the original image to the CNN. Instead, it uses a classical image processing filter to create an "artifact map" that isolates the contaminant. The CNN is then trained *only* on this map.

This hybrid approach forces the model to learn the correct features.

### Step 1: The "Difference of Gaussians" (DoG) Pre-processing Filter

The key insight is that the "noise" (wrinkles, shadows) and the "signal" (hair, particles) exist in different spatial frequencies.

  * **Wrinkles/Shadows:** Low-frequency (broad, soft, blurry shapes).
  * **Hair/Particles:** High-frequency (thin, sharp, fine-detailed lines and specks).[1]

We use a **Difference of Gaussians (DoG)** filter to act as a band-pass filter, separating these frequencies. This technique is also a computationally efficient approximation of the Laplacian of Gaussian (LoG) operator, which is excellent for blob and edge detection.

The process is as follows:

1.  **Read Image:** The $1920\times1080$ color frame is read from the webcam.
2.  **Grayscale:** The image is converted to grayscale.
3.  **Create Broad Blur:** We apply a large Gaussian blur (e.g., $25\times25$ kernel). This "blurs out" the fine details of the hair and particles, leaving *only* the low-frequency wrinkles and shadows.
4.  **Create Fine Blur:** We apply a very small Gaussian blur (e.g., $3\times3$ kernel). This removes tiny pixel noise but preserves the hair, particles, *and* the wrinkles.
5.  **Subtract:** We subtract the `Broad Blur` image from the `Fine Blur` image. This subtraction cancels out the common low-frequency information (the wrinkles/shadows), leaving *only* the high-frequency details.
6.  **Threshold & Normalize:** The resulting "artifact map" is thresholded to make the contaminants stand out as white pixels on a black background.

**Visualizing the Process:**

| Original Image (High-Noise) | DoG Filter Output (High-Signal) |
| :--- | :--- |
|\!([httpsfakesite.com/wrinkled\_drape.png](https://www.google.com/search?q=https://httpsfakesite.com/wrinkled_drape.png)) | |
| **Result:** A *sterile* drape becomes a (mostly) black image. A *contaminated* drape becomes a black image with clear white specks or lines. |

### Step 2: The Simple CNN Classifier

The "complexity" is now handled. The CNN's job is simple. It is *not* trained on the original photos, but *exclusively* on the `DoG_Map` outputs from Step 1.

  * **Architecture:** A lightweight, custom CNN (e.g., 4-5 convolutional layers followed by a dense classifier). This avoids the need for heavy, pre-trained models.
  * **Input:** A $128\times128\times1$ (grayscale) `DoG_Map` image.
  * **Output:** A 4-class softmax classification:
    1.  `sterile`
    2.  `contaminant_hair`
    3.  `contaminant_trash`
    4.  `contaminant_both`

The network learns to classify the *patterns* of the artifacts. A long, thin line is `hair` [2], a small cluster of specks is `trash`, and a black image is `sterile`.

## Project Structure

```
/sterile-field-detector
│
├── 01_data_collection/
│   ├── capture_video.py        # Simple script to record.mp4 files for each class.
│   └── extract_frames.py       # Converts videos to frames (e.g., 4fps) and saves to /dataset_raw.
│
├── 02_preprocessing/
│   ├── build_dataset.py        # Applies the DoG filter to /dataset_raw and saves maps to /dataset_processed.
│
├── 03_training/
│   ├── model.py                # Defines the simple CNN architecture.
│   ├── train.py                # Loads processed data and trains the model.
│   └── sterile_model.h5        # The final trained model weights.
│
├── 04_inference/
│   └── run_live_detector.py    # Runs the full pipeline (capture -> filter -> CNN) on a live webcam feed.
│
└── README.md                   # You are here.
```

## Dataset Generation (Zero-Cost)

The dataset can be generated in under 15 minutes using a webcam and basic household items.

**Materials:**

  * **Webcam:** Any standard webcam.
  * **Sterile Drape:** A blue paper shop towel, a piece of craft paper, or any solid-color (blue/green) cloth that wrinkles easily.
  * **Contaminants:**
      * `hair`: A single human hair.
      * `trash`: A few tiny scraps of paper, salt/sugar granules, or breadcrumbs.
  * **Environment:** A desk lamp to create harsh shadows and wrinkles.

**Process:**

1.  Set up your "drape" and wrinkle it. Position the lamp to create strong shadows.
2.  Run `01_data_collection/capture_video.py`.
3.  Record 5-10 seconds of video for each of the four classes:
      * **Class 1 (`sterile`):** Record the drape with only wrinkles and shadows. Move the camera slightly.
      * **Class 2 (`contaminant_hair`):** Drop the hair onto the drape. Record it in different positions.
      * **Class 3 (`contaminant_trash`):** Place the small scraps on the drape. Record.
      * **Class 4 (`contaminant_both`):** Add both the hair and the scraps. Record.
4.  Run `01_data_collection/extract_frames.py`. This script will pull \~2-4 frames per second from your videos and create a raw dataset of 1,000-2,000 images, automatically sorted into `sterile/`, `hair/`, etc.

## How to Run

**1. Create the Dataset:**
Follow the steps in **Dataset Generation** above.

**2. Pre-process the Data:**
Run the DoG filter script. This converts all raw frames into artifact maps.

```bash
python 02_preprocessing/build_dataset.py
```

**3. Train the Model:**
Run the training script. This will load the processed maps and train the simple CNN.

```bash
python 03_training/train.py
```

**4. Run the Live Detector:**
Run the inference script to see the live results from your webcam\!

```bash
python 04_inference/run_live_detector.py
```

## Technologies Used

  * **Python 3.10+**
  * **OpenCV (`opencv-python`)**: Used for all image capture, video processing, and classical filtering (Grayscale, GaussianBlur, Subtract, Threshold).
  * **TensorFlow / Keras:** Used to build, train, and run the simple CNN classifier.
  * **Numpy:** For numerical operations.

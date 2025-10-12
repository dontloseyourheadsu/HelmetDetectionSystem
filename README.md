# HelmetDetectionSystem

## Computer Vision-Based Security Application

### Project Overview
This project implements an intelligent helmet detection and classification system capable of distinguishing between bicycle helmets, motorcycle helmets, and construction helmets. Unlike traditional binary detection systems, this solution provides multi-class classification with hierarchical detection methodology optimized for resource-constrained environments.

**Key Features:**
- Multi-class helmet classification (bicycle, motorcycle, construction)
- Hierarchical detection pipeline (head detection → helmet verification → type classification)
- Optimized for low-resource deployment
- Real-time processing capabilities (~10 FPS on NVIDIA T4 GPU)
- Privacy-focused architecture with minimal data retention

---

## Technologies & Architecture

### Core Technologies
- **Deep Learning Framework:** PyTorch 2.0
- **Object Detection:** YOLOv8 (Ultralytics implementation)
- **Classification Models:** 
  - EfficientNet-B3 (primary classifier)
  - ResNet-50 (feature extraction)
  - MobileNetV2 (lightweight alternative)
- **Computer Vision:** OpenCV 4.x
- **UI Framework:** GTK 3.0 / PyQt5 (Linux Desktop Application)
- **Training Environment:** Google Colab (Free Tier - NVIDIA T4 GPU)
- **Experiment Tracking:** Weights & Biases
- **Mixed Precision Training:** PyTorch AMP (FP16)

### Feature Engineering
- **Traditional CV Techniques:**
  - Histogram of Oriented Gradients (HOG) for shape features
  - Local Binary Patterns (LBP) for texture analysis
  - Gaussian filtering (3x3 kernel) for noise reduction
  - Histogram equalization for contrast enhancement
  - Morphological operations (dilation/erosion) for region refinement

---

## System Architecture

### 1. Model Training Components (Google Colab)

#### a) Data Preprocessing Pipeline
```
Raw Images → Format Standardization → Quality Filtering → Augmentation
     ↓
Gaussian Filtering → Histogram Equalization → Region Extraction
     ↓
Annotation Verification → Dataset Partitioning (70/15/15)
```

**Preprocessing Steps:**
- Image resizing and normalization
- Noise reduction with Gaussian filtering (3x3 kernel)
- Contrast enhancement via histogram equalization
- Morphological operations for artifact removal
- Privacy-preserving facial blurring (when applicable)

#### b) Hierarchical Detection Model
```
Input Image
    ↓
[Stage 1: Head Detection - YOLOv8]
    ↓
Head Region Extraction
    ↓
[Stage 2: Helmet Verification - Binary Classifier]
    ↓
Helmet Region
    ↓
[Stage 3: Helmet Classification - EfficientNet-B3]
    ↓
Output: {bicycle, motorcycle, construction, no-helmet}
```

#### c) Feature Extraction
- **Transfer Learning:** Pre-trained weights from ImageNet
- **HOG Features:** Shape and edge information
- **LBP Features:** Texture patterns for material distinction
- **Deep Features:** Extracted from intermediate CNN layers

#### d) Training Optimization
- Progressive training with increasing image resolution
- Mixed precision (FP16) for memory efficiency
- Gradient accumulation for effective large batch sizes
- Synthetic oversampling (SMOTE) for class balancing
- Negative examples integration for robustness

### 2. Linux Desktop UI Application

#### Application Architecture
```
┌─────────────────────────────────────────┐
│         GTK/PyQt Main Window            │
├─────────────────────────────────────────┤
│  Camera/Video Input   │   Results Panel │
│  ┌─────────────────┐  │  ┌───────────┐  │
│  │  Live Feed      │  │  │ Helmet    │  │
│  │  or File Upload │  │  │ Type:     │  │
│  │                 │  │  │ Confidence│  │
│  └─────────────────┘  │  └───────────┘  │
│                       │                  │
│  Control Panel        │   Statistics     │
│  [Start] [Stop]       │   Dashboard      │
│  [Load Model]         │                  │
└─────────────────────────────────────────┘
```

**UI Features:**
- Real-time video/webcam processing
- Static image analysis
- Batch processing capabilities
- Confidence score visualization
- Detection history and statistics
- Model selection interface
- Privacy controls (auto-blur, local processing)

#### UI-Model Integration
- Loads trained model weights from Colab export
- ONNX runtime for optimized inference
- Preprocessing pipeline implementation
- Post-processing and visualization
- Threading for non-blocking UI

---

## Development Roadmap

### Phase 1: Data Preparation (Weeks 1-2)
- [ ] Dataset acquisition and licensing verification
- [ ] Preprocessing pipeline implementation
- [ ] Data augmentation strategy
- [ ] Train/validation/test split (70/15/15)
- [ ] Negative examples collection

### Phase 2: Model Development (Weeks 3-5)
- [ ] YOLOv8 head detector training
- [ ] Helmet verification classifier
- [ ] Multi-class classifier (EfficientNet-B3)
- [ ] Feature extraction optimization
- [ ] Cascade integration and testing

### Phase 3: UI Development (Weeks 4-6)
- [ ] Desktop application framework setup
- [ ] Camera/video input integration
- [ ] Model inference pipeline
- [ ] Results visualization
- [ ] User controls and settings

### Phase 4: Evaluation & Optimization (Weeks 7-8)
- [ ] Performance metrics evaluation (mAP, accuracy)
- [ ] Inference speed optimization (<100ms target)
- [ ] Model quantization and pruning
- [ ] Bias analysis across demographics
- [ ] Documentation and deployment guide

---

## Installation & Setup

### Prerequisites
- **For Training (Google Colab):**
  - Google account with Colab access
  - Google Drive (for dataset storage)
  
- **For UI Application (Linux):**
  - Ubuntu 20.04+ / Debian 11+ / Fedora 35+
  - Python 3.8+
  - CUDA 11.8+ (for GPU acceleration, optional)
  - 4GB RAM minimum (8GB recommended)

### 1. Training Environment Setup (Google Colab)

#### Step 1: Clone Repository
```bash
# In a Colab notebook cell
!git clone https://github.com/your-username/helmet-detection.git
%cd helmet-detection
```

#### Step 2: Install Dependencies
```python
# Install required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install ultralytics==8.0.196
!pip install efficientnet-pytorch
!pip install opencv-python-headless
!pip install albumentations
!pip install wandb
!pip install scikit-learn
!pip install pandas matplotlib seaborn
```

#### Step 3: Dataset Setup

**Option A: Download Pre-configured Datasets**
```python
# Create data directory structure
!mkdir -p data/{raw,processed,augmented}

# Download datasets (examples - replace with actual URLs)
from roboflow import Roboflow

# Safety Helmet Detection (CC BY 4.0)
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("safety-helmets").project("helmet-detection")
dataset = project.version(1).download("yolov8")

# Bike Helmet Detection (MIT License)
!kaggle datasets download -d andrewmvd/bike-helmet-detection
!unzip bike-helmet-detection.zip -d data/raw/bike_helmets/

# Construction Helmet (Apache 2.0)
!wget https://example.com/construction-helmets.zip
!unzip construction-helmets.zip -d data/raw/construction/

# Hard Hat Workers (CC0 1.0)
!wget https://example.com/hard-hat-workers.zip
!unzip hard-hat-workers.zip -d data/raw/hard_hat/
```

**Option B: Manual Dataset Organization**
```
data/
├── raw/
│   ├── bike_helmets/
│   │   ├── images/
│   │   └── labels/
│   ├── motorcycle_helmets/
│   │   ├── images/
│   │   └── labels/
│   ├── construction_helmets/
│   │   ├── images/
│   │   └── labels/
│   └── negative_examples/
│       ├── images/
│       └── labels/
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
└── augmented/
```

**Dataset Format Requirements:**
- Images: JPG/PNG format
- Labels: YOLO format (.txt files)
  ```
  class_id center_x center_y width height
  # Example: 0 0.5 0.5 0.3 0.4
  ```
- Class mapping:
  ```
  0: bicycle_helmet
  1: motorcycle_helmet
  2: construction_helmet
  3: no_helmet
  ```

#### Step 4: Run Preprocessing
```python
# Run preprocessing pipeline
!python src/preprocessing/preprocess.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --img_size 640 \
  --apply_gaussian \
  --apply_histogram_eq \
  --train_split 0.7 \
  --val_split 0.15 \
  --test_split 0.15
```

#### Step 5: Train Models
```python
# Train YOLOv8 head detector
!python src/training/train_head_detector.py \
  --data data/processed/head_detection.yaml \
  --epochs 100 \
  --batch 16 \
  --img 640

# Train helmet classifier
!python src/training/train_classifier.py \
  --model efficientnet-b3 \
  --data data/processed \
  --epochs 50 \
  --batch 32 \
  --lr 0.001 \
  --mixed_precision
```

#### Step 6: Export Models
```python
# Export to ONNX for deployment
!python src/export/export_models.py \
  --head_detector runs/detect/train/weights/best.pt \
  --classifier runs/classify/train/weights/best.pt \
  --output models/exported/
```

### 2. Linux Desktop UI Setup

#### Step 1: Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-pip python3-venv
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-3.0
sudo apt install libcairo2-dev libgirepository1.0-dev
sudo apt install v4l-utils  # For webcam support

# Fedora
sudo dnf install python3-pip python3-virtualenv
sudo dnf install python3-gobject gtk3-devel
sudo dnf install cairo-devel gobject-introspection-devel
```

#### Step 2: Create Virtual Environment
```bash
cd helmet-detection-ui
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt:**
```
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.0.76
onnxruntime==1.15.1
PyGObject==3.44.1
numpy==1.24.3
Pillow==10.0.0
matplotlib==3.7.2
```

#### Step 4: Download Pre-trained Models
```bash
# Create models directory
mkdir -p models

# Download from Google Drive (after Colab training)
# Replace FILE_ID with your shared model file ID
gdown https://drive.google.com/uc?id=FILE_ID -O models/head_detector.onnx
gdown https://drive.google.com/uc?id=FILE_ID -O models/classifier.onnx

# Or use wget if hosted elsewhere
wget https://your-storage.com/models.zip
unzip models.zip -d models/
```

#### Step 5: Configure Application
```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit configuration
nano config.yaml
```

**config.yaml:**
```yaml
model:
  head_detector: "models/head_detector.onnx"
  classifier: "models/classifier.onnx"
  confidence_threshold: 0.7
  iou_threshold: 0.5

preprocessing:
  image_size: 640
  gaussian_kernel: 3
  enable_histogram_eq: true

ui:
  theme: "dark"
  show_confidence: true
  enable_privacy_mode: true
  fps_limit: 30

inference:
  device: "cuda"  # or "cpu"
  batch_size: 1
  num_threads: 4
```

#### Step 6: Run Application
```bash
# Activate virtual environment
source venv/bin/activate

# Run UI application
python3 src/ui/main.py

# Or with specific config
python3 src/ui/main.py --config config.yaml
```

---

## Usage Guide

### Training New Models

1. **Prepare your dataset** following the structure above
2. **Open Colab notebook:** `notebooks/train_model.ipynb`
3. **Mount Google Drive** and upload datasets
4. **Run preprocessing cells** to prepare data
5. **Execute training cells** with desired hyperparameters
6. **Monitor training** via Weights & Biases dashboard
7. **Export models** when training completes
8. **Download models** to local machine

### Using the Desktop Application

1. **Launch application:** `python3 src/ui/main.py`
2. **Load models:** File → Load Models → Select ONNX files
3. **Select input source:**
   - Camera: Select webcam device
   - Video: Load video file
   - Image: Load single/multiple images
4. **Start detection:** Click "Start" button
5. **View results:** Real-time classification with bounding boxes
6. **Export results:** Save annotated images/videos

### API Usage (Optional)

```python
from helmet_detector import HelmetDetector

# Initialize detector
detector = HelmetDetector(
    head_model='models/head_detector.onnx',
    classifier_model='models/classifier.onnx'
)

# Process image
result = detector.detect('image.jpg')
print(f"Helmet Type: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}")

# Process video
detector.process_video('video.mp4', output='output.mp4')
```

---

## Performance Metrics

### Target Metrics
- **mAP@0.5:** ≥ 0.75
- **Classification Accuracy:** ≥ 85% across all classes
- **Inference Speed:** ≤ 100ms per image (T4 GPU)
- **False Positive Rate:** < 5%
- **Demographic Variance:** < 10% accuracy difference

### Evaluation
```bash
# Run evaluation script
python src/evaluation/evaluate.py \
  --model_dir models/exported \
  --test_data data/processed/test \
  --output_dir results/evaluation
```

---

## Project Structure

```
helmet-detection/
├── notebooks/
│   ├── train_model.ipynb          # Main training notebook
│   ├── data_exploration.ipynb     # Dataset analysis
│   └── model_evaluation.ipynb     # Performance analysis
├── src/
│   ├── preprocessing/
│   │   ├── preprocess.py          # Data preprocessing pipeline
│   │   ├── augmentation.py        # Data augmentation
│   │   └── feature_extraction.py  # HOG, LBP features
│   ├── training/
│   │   ├── train_head_detector.py
│   │   ├── train_classifier.py
│   │   └── utils.py
│   ├── models/
│   │   ├── head_detector.py       # YOLOv8 wrapper
│   │   ├── classifier.py          # EfficientNet classifier
│   │   └── cascade.py             # Hierarchical pipeline
│   ├── export/
│   │   └── export_models.py       # ONNX conversion
│   ├── ui/
│   │   ├── main.py                # Application entry point
│   │   ├── gui.py                 # GTK interface
│   │   ├── camera.py              # Video capture
│   │   ├── inference.py           # Model inference
│   │   └── visualization.py       # Result rendering
│   └── evaluation/
│       ├── evaluate.py
│       └── metrics.py
├── data/                          # (Not tracked in git)
│   ├── raw/
│   ├── processed/
│   └── augmented/
├── models/                        # (Not tracked in git)
│   └── exported/
├── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory (Colab)**
```python
# Reduce batch size
--batch 8

# Enable gradient accumulation
--accumulate 4

# Use mixed precision
--mixed_precision
```

**2. UI Application Won't Start**
```bash
# Install missing GTK dependencies
sudo apt install gir1.2-gtk-3.0

# Check Python GTK bindings
python3 -c "import gi; gi.require_version('Gtk', '3.0')"
```

**3. Webcam Not Detected**
```bash
# List available cameras
v4l2-ctl --list-devices

# Grant permissions
sudo usermod -a -G video $USER
```

**4. Slow Inference**
- Ensure ONNX Runtime with GPU support is installed
- Check CUDA availability: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Reduce input image resolution in config.yaml

---

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Follow PEP 8 style guidelines
4. Add tests for new features
5. Submit pull request with detailed description

---

## License

This project uses multiple components with different licenses:
- **Code:** MIT License
- **Datasets:** Various (see dataset table above)
- **Pre-trained Models:** Respective model licenses

---

## Acknowledgments

- **Datasets:** Roboflow Universe, Kaggle community
- **Pre-trained Models:** PyTorch, Ultralytics, EfficientNet authors
- **Infrastructure:** Google Colab, Weights & Biases

---

## Contact & Support

- **Author:** Jesus Alvarez Sombrerero (177516)
- **Course:** Artificial Vision LIS 4042, Section 1
- **Issues:** GitHub Issues page
- **Documentation:** See `docs/` directory

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{helmet_detection_2025,
  author = {Alvarez Sombrerero, Jesus},
  title = {Multi-Class Helmet Detection System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-username/helmet-detection}
}
```

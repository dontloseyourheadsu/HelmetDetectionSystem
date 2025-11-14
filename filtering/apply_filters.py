import cv2
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---

# 1) Kernel Sizes (ODD numbers)
SMALL_KERNEL_SIZE = (5, 5)    # preserves thin lines
LARGE_KERNEL_SIZE = (25, 25)  # removes low-frequency background

# 2) Thresholding
THRESHOLD_VALUE = 15

# 3) Shape Classification Parameters (NEW: from advanced algorithm)
# These parameters help distinguish hair from trash based on shape
SHAPE_PARAMS = {
    # --- Hair (Elongated, Solid) ---
    'hair_min_area': 15,          # Min pixels to be considered
    'hair_max_area': 2000,        # Max pixels (filters large shadows)
    'hair_min_eccentricity': 4.0, # Higher = more "line-like" (MajorAxis / MinorAxis)
    'hair_min_solidity': 0.75,    # Ratio of contour area to its convex hull area

    # --- Trash (Compact, Solid) ---
    'trash_min_area': 5,
    'trash_max_area': 200,
    'trash_max_eccentricity': 2.5, # Lower = more "blob-like"
    'trash_min_solidity': 0.85
}

# 4) Output size
TARGET_IMG_SIZE = (128, 128)

# 5) Output mode: 'grayscale' or 'multichannel'
# 'multichannel' creates a 3-channel BGR image: B=Cloth, G=Trash, R=Hair
OUTPUT_MODE = 'multichannel'  # Change to 'grayscale' for original behavior

# 6) Directories (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = (SCRIPT_DIR / "../dataset/frames").resolve()
# Per user request: processed folder must be relative to filtering folder
PROCESSED_DATA_DIR = (SCRIPT_DIR / "processed").resolve()
# Required cloth color reference image (relative to this script)
CLOTH_REF_PATH = (SCRIPT_DIR / "tablecloth-color.png").resolve()

# 7) Cloth color range deltas (HSV)
H_DELTA = 12
S_DELTA = 80
V_DELTA = 80

# ---------------------------------------------------------------


def list_classes(root: Path) -> List[str]:
	classes: List[str] = []
	if not root.exists():
		return classes
	for p in sorted(root.iterdir()):
		if p.is_dir() and not p.name.startswith('.'):
			classes.append(p.name)
	return classes


def clamp(val: int, lo: int, hi: int) -> int:
	return max(lo, min(hi, val))



def hsv_range_from_reference(ref_img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	ref_bgr = cv2.imread(str(ref_img_path))
	if ref_bgr is None:
		raise FileNotFoundError(f"Could not read cloth reference image: {ref_img_path}")
	ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)
	h = int(np.mean(ref_hsv[..., 0]))
	s = int(np.mean(ref_hsv[..., 1]))
	v = int(np.mean(ref_hsv[..., 2]))
	lower = np.array([
		clamp(h - H_DELTA, 0, 179),
		clamp(s - S_DELTA, 0, 255),
		clamp(v - V_DELTA, 0, 255),
	], dtype=np.uint8)
	upper = np.array([
		clamp(h + H_DELTA, 0, 179),
		clamp(s + S_DELTA, 0, 255),
		clamp(v + V_DELTA, 0, 255),
	], dtype=np.uint8)
	return lower, upper


def crop_to_cloth_region(img_bgr: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray,
						 margin_ratio: float = 0.05) -> np.ndarray:
	"""Detect cloth by color, crop a tight bounding rectangle with a small margin.
	Falls back to original image if no region is detected.
	"""
	hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
	# Clean mask
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return img_bgr

	c = max(contours, key=cv2.contourArea)
	if cv2.contourArea(c) < 500:  # too small
		return img_bgr

	x, y, w, h = cv2.boundingRect(c)
	H, W = img_bgr.shape[:2]
	pad_x = int(w * margin_ratio)
	pad_y = int(h * margin_ratio)
	x0 = clamp(x - pad_x, 0, W - 1)
	y0 = clamp(y - pad_y, 0, H - 1)
	x1 = clamp(x + w + pad_x, 0, W)
	y1 = clamp(y + h + pad_y, 0, H)
	roi = img_bgr[y0:y1, x0:x1]
	return roi if roi.size > 0 else img_bgr


def classify_contours_by_shape(
	contours: List[np.ndarray],
	params: Dict[str, float]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
	"""
	Analyzes contours and classifies them as 'hair' or 'trash' based on shape properties.
	Uses eccentricity (elongation) and solidity (compactness) metrics.
	
	Returns:
		(hair_contours, trash_contours)
	"""
	hair_contours = []
	trash_contours = []
	
	for c in contours:
		area = cv2.contourArea(c)
		
		# Need >= 5 points to fit an ellipse
		if len(c) < 5:
			continue
			
		try:
			# Calculate Solidity (how "solid" vs "irregular" the shape is)
			hull = cv2.convexHull(c)
			hull_area = cv2.contourArea(hull)
			if hull_area == 0:
				continue
			solidity = float(area) / hull_area
			
			# Calculate Eccentricity (elongation: major axis / minor axis)
			(x, y), (MA, ma), angle = cv2.fitEllipse(c)
			if ma == 0:
				continue
			eccentricity = float(MA) / ma  # High value = elongated (hair-like)
			
			# --- Hair Logic: Elongated and solid ---
			if (params['hair_min_area'] < area < params['hair_max_area'] and
				eccentricity > params['hair_min_eccentricity'] and 
				solidity > params['hair_min_solidity']):
				hair_contours.append(c)
				
			# --- Trash Logic: Compact and solid ---
			elif (params['trash_min_area'] < area < params['trash_max_area'] and
				  eccentricity < params['trash_max_eccentricity'] and 
				  solidity > params['trash_min_solidity']):
				trash_contours.append(c)
				
		except cv2.error:
			continue  # cv2.fitEllipse can fail on degenerate contours
			
	return hair_contours, trash_contours


def enhance_and_filter(gray: np.ndarray) -> np.ndarray:
	"""Enhance features and apply DoG + threshold to highlight thin hairs and crumbs."""
	# CLAHE to boost contrast
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	gray = clahe.apply(gray)

	# DoG
	small_blur = cv2.GaussianBlur(gray, SMALL_KERNEL_SIZE, 0)
	large_blur = cv2.GaussianBlur(gray, LARGE_KERNEL_SIZE, 0)
	dog = cv2.subtract(small_blur, large_blur)

	# Edge emphasis (Sobel magnitude) and combine
	sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
	sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
	sobel_mag = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(sobelx), 1.0,
													cv2.convertScaleAbs(sobely), 1.0, 0))
	combined = cv2.addWeighted(dog, 0.7, sobel_mag, 0.3, 0)

	# Binary map
	_, thresh = cv2.threshold(combined, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# Light morphological smoothing to connect hair lines, keep crumbs
	kernel = np.ones((3, 3), np.uint8)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
	return thresh


def create_multichannel_output(
	h: int, w: int,
	cloth_mask: np.ndarray,
	hair_contours: List[np.ndarray],
	trash_contours: List[np.ndarray]
) -> np.ndarray:
	"""
	Creates a 3-channel BGR feature image for CNN training:
	- Blue Channel = Cloth Area (cleaned background)
	- Green Channel = Trash Mask
	- Red Channel = Hair Mask
	"""
	# Create blank masks for hair and trash
	mask_hair = np.zeros((h, w), dtype=np.uint8)
	mask_trash = np.zeros((h, w), dtype=np.uint8)
	
	# Draw the approved contours (filled)
	if hair_contours:
		cv2.drawContours(mask_hair, hair_contours, -1, 255, cv2.FILLED)
	if trash_contours:
		cv2.drawContours(mask_trash, trash_contours, -1, 255, cv2.FILLED)
	
	# Merge into BGR: B=Cloth, G=Trash, R=Hair
	output_image = cv2.merge((cloth_mask, mask_trash, mask_hair))
	
	# Resize to target size
	output_resized = cv2.resize(output_image, TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)
	
	return output_resized


def process_image(image_path: Path, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> np.ndarray | None:
	bgr = cv2.imread(str(image_path))
	if bgr is None:
		print(f"Warning: Could not read image {image_path}. Skipping.")
		return None

	# 1) Crop to cloth region using color mask
	roi_bgr = crop_to_cloth_region(bgr, lower_hsv, upper_hsv, margin_ratio=0.06)
	h, w = roi_bgr.shape[:2]

	# 2) Convert to gray and enhance
	gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
	filtered = enhance_and_filter(gray)

	# 3) Choose output mode
	if OUTPUT_MODE == 'multichannel':
		# NEW: Multi-channel approach with shape-based classification
		# Create a cloth mask (inverted filtered to show cloth area)
		cloth_mask = cv2.bitwise_not(filtered)
		
		# Find all contours in the filtered image
		contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		# Classify contours by shape
		hair_contours, trash_contours = classify_contours_by_shape(contours, SHAPE_PARAMS)
		
		# Create multi-channel output
		out = create_multichannel_output(h, w, cloth_mask, hair_contours, trash_contours)
	else:
		# Original: Single-channel grayscale output
		out = cv2.resize(filtered, TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)
	
	return out


def find_first_image(root: Path) -> Path | None:
	exts = {'.png', '.jpg', '.jpeg', '.bmp'}
	classes = list_classes(root)
	for cls in classes:
		cls_dir = root / cls
		for p in sorted(cls_dir.iterdir()):
			if p.suffix.lower() in exts:
				return p
	return None


def estimate_hsv_range_from_sample(sample_img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	bgr = cv2.imread(str(sample_img_path))
	if bgr is None:
		raise FileNotFoundError(f"Could not read sample image to estimate cloth color: {sample_img_path}")
	H, W = bgr.shape[:2]
	# Use central region to approximate the cloth color
	x0, x1 = int(W * 0.25), int(W * 0.75)
	y0, y1 = int(H * 0.25), int(H * 0.75)
	center = bgr[y0:y1, x0:x1]
	hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
	# Slight blur to reduce noise before statistics
	hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
	h = int(np.median(hsv[..., 0]))
	s = int(np.median(hsv[..., 1]))
	v = int(np.median(hsv[..., 2]))
	lower = np.array([
		clamp(h - H_DELTA, 0, 179),
		clamp(s - S_DELTA, 0, 255),
		clamp(v - V_DELTA, 0, 255),
	], dtype=np.uint8)
	upper = np.array([
		clamp(h + H_DELTA, 0, 179),
		clamp(s + S_DELTA, 0, 255),
		clamp(v + V_DELTA, 0, 255),
	], dtype=np.uint8)
	return lower, upper


def main() -> None:
	# Prefer cloth reference if present; otherwise estimate from first available image
	if CLOTH_REF_PATH.exists():
		lower_hsv, upper_hsv = hsv_range_from_reference(CLOTH_REF_PATH)
		print(f"Using cloth reference: {CLOTH_REF_PATH}")
	else:
		sample = find_first_image(RAW_DATA_DIR)
		if sample is None:
			print(f"Error: No images under {RAW_DATA_DIR} to estimate cloth color, and reference file is missing.")
			print(f"Place a cloth reference at {CLOTH_REF_PATH} and rerun.")
			raise SystemExit(2)
		print(f"Reference not found. Estimating cloth color from sample: {sample}")
		lower_hsv, upper_hsv = estimate_hsv_range_from_sample(sample)

	print("Starting dataset preprocessing with cloth-aware zoom…")
	print(f"Output mode:    {OUTPUT_MODE.upper()}")
	if OUTPUT_MODE == 'multichannel':
		print(f"  └─ Using shape-based classification (Hair vs Trash)")
	print(f"Reading from:   {RAW_DATA_DIR}")
	print(f"Saving maps to: {PROCESSED_DATA_DIR}")
	print("-" * 30)

	processed_count = 0
	PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

	classes = list_classes(RAW_DATA_DIR)
	if not classes:
		print(f"No class folders found under {RAW_DATA_DIR}. Nothing to process.")
		return

	for class_name in classes:
		input_class_dir = RAW_DATA_DIR / class_name
		output_class_dir = PROCESSED_DATA_DIR / class_name
		output_class_dir.mkdir(parents=True, exist_ok=True)

		exts = {'.png', '.jpg', '.jpeg', '.bmp'}
		image_files = [p for p in sorted(input_class_dir.iterdir()) if p.suffix.lower() in exts]
		if not image_files:
			print(f"Warning: No images found for class '{class_name}' in {input_class_dir}. Skipping.")
			continue

		print(f"Processing {len(image_files)} images for class: '{class_name}'")
		for image_path in tqdm(image_files, desc=class_name, unit="img"):
			out_path = output_class_dir / (image_path.stem + ".png")
			# Skip if already processed
			if out_path.exists():
				continue
			result = process_image(image_path, lower_hsv, upper_hsv)
			if result is None:
				continue
			cv2.imwrite(str(out_path), result)
			processed_count += 1

	print("-" * 30)
	print(f"Preprocessing complete. Total images processed: {processed_count}")
	print(f"Processed dataset is ready in: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
	main()


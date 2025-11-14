import cv2
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

SMALL_KERNEL_SIZE = (5, 5)
LARGE_KERNEL_SIZE = (25, 25)
THRESHOLD_VALUE = 15

SHAPE_PARAMS = {
    'hair_min_area': 15,
    'hair_max_area': 2000,
    'hair_min_eccentricity': 4.0,
    'hair_min_solidity': 0.75,
    'trash_min_area': 5,
    'trash_max_area': 200,
    'trash_max_eccentricity': 2.5,
    'trash_min_solidity': 0.85
}

TARGET_IMG_SIZE = (128, 128)
OUTPUT_MODE = 'multichannel'

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = (SCRIPT_DIR / "../dataset/frames").resolve()
PROCESSED_DATA_DIR = (SCRIPT_DIR / "processed").resolve()
CLOTH_REF_PATH = (SCRIPT_DIR / "tablecloth-color.png").resolve()

H_DELTA = 12
S_DELTA = 80
V_DELTA = 80


def list_classes(root: Path) -> List[str]:
	"""List all class directories in the given root path.
	
	Args:
		root: Path to the root directory containing class folders.
		
	Returns:
		List of class directory names, excluding hidden directories.
	"""
	classes: List[str] = []
	if not root.exists():
		return classes
	for p in sorted(root.iterdir()):
		if p.is_dir() and not p.name.startswith('.'):
			classes.append(p.name)
	return classes


def clamp(val: int, lo: int, hi: int) -> int:
	"""Clamp a value between a lower and upper bound.
	
	Args:
		val: Value to clamp.
		lo: Lower bound.
		hi: Upper bound.
		
	Returns:
		The clamped value.
	"""
	return max(lo, min(hi, val))



def hsv_range_from_reference(ref_img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	"""Calculate HSV color range from a reference cloth image.
	
	Args:
		ref_img_path: Path to the reference cloth image.
		
	Returns:
		Tuple of (lower_bound, upper_bound) HSV arrays.
		
	Raises:
		FileNotFoundError: If the reference image cannot be read.
	"""
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
	"""Detect cloth by color and crop to a bounding rectangle with margin.
	
	Args:
		img_bgr: Input BGR image.
		lower_hsv: Lower bound HSV array for cloth color.
		upper_hsv: Upper bound HSV array for cloth color.
		margin_ratio: Margin ratio to add around detected cloth region.
		
	Returns:
		Cropped image containing the cloth region, or original image if no region is detected.
	"""
	hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return img_bgr

	c = max(contours, key=cv2.contourArea)
	if cv2.contourArea(c) < 500:
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
	"""Classify contours as hair or trash based on shape properties.
	
	Uses eccentricity (elongation) and solidity (compactness) metrics to distinguish
	between elongated hair-like objects and compact trash-like objects.
	
	Args:
		contours: List of contours to classify.
		params: Dictionary of shape classification parameters.
		
	Returns:
		Tuple of (hair_contours, trash_contours).
	"""
	hair_contours = []
	trash_contours = []
	
	for c in contours:
		area = cv2.contourArea(c)
		
		if len(c) < 5:
			continue
			
		try:
			hull = cv2.convexHull(c)
			hull_area = cv2.contourArea(hull)
			if hull_area == 0:
				continue
			solidity = float(area) / hull_area
			
			(x, y), (MA, ma), angle = cv2.fitEllipse(c)
			if ma == 0:
				continue
			eccentricity = float(MA) / ma
			
			if (params['hair_min_area'] < area < params['hair_max_area'] and
				eccentricity > params['hair_min_eccentricity'] and 
				solidity > params['hair_min_solidity']):
				hair_contours.append(c)
				
			elif (params['trash_min_area'] < area < params['trash_max_area'] and
				  eccentricity < params['trash_max_eccentricity'] and 
				  solidity > params['trash_min_solidity']):
				trash_contours.append(c)
				
		except cv2.error:
			continue
			
	return hair_contours, trash_contours


def enhance_and_filter(gray: np.ndarray) -> np.ndarray:
	"""Enhance image features using DoG and edge detection.
	
	Applies CLAHE contrast enhancement, Difference of Gaussians (DoG), and Sobel edge
	detection to highlight thin contaminants like hairs and crumbs.
	
	Args:
		gray: Input grayscale image.
		
	Returns:
		Binary thresholded image with enhanced features.
	"""
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	gray = clahe.apply(gray)

	small_blur = cv2.GaussianBlur(gray, SMALL_KERNEL_SIZE, 0)
	large_blur = cv2.GaussianBlur(gray, LARGE_KERNEL_SIZE, 0)
	dog = cv2.subtract(small_blur, large_blur)

	sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
	sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
	sobel_mag = cv2.convertScaleAbs(cv2.addWeighted(cv2.convertScaleAbs(sobelx), 1.0,
													cv2.convertScaleAbs(sobely), 1.0, 0))
	combined = cv2.addWeighted(dog, 0.7, sobel_mag, 0.3, 0)

	_, thresh = cv2.threshold(combined, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	kernel = np.ones((3, 3), np.uint8)
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
	return thresh


def create_multichannel_output(
	h: int, w: int,
	cloth_mask: np.ndarray,
	hair_contours: List[np.ndarray],
	trash_contours: List[np.ndarray]
) -> np.ndarray:
	"""Create a 3-channel BGR feature image for CNN training.
	
	Channels represent:
	- Blue: Cloth area (cleaned background)
	- Green: Trash mask
	- Red: Hair mask
	
	Args:
		h: Height of the output image.
		w: Width of the output image.
		cloth_mask: Binary mask of the cloth area.
		hair_contours: List of contours classified as hair.
		trash_contours: List of contours classified as trash.
		
	Returns:
		Resized 3-channel BGR image with encoded features.
	"""
	mask_hair = np.zeros((h, w), dtype=np.uint8)
	mask_trash = np.zeros((h, w), dtype=np.uint8)
	
	if hair_contours:
		cv2.drawContours(mask_hair, hair_contours, -1, 255, cv2.FILLED)
	if trash_contours:
		cv2.drawContours(mask_trash, trash_contours, -1, 255, cv2.FILLED)
	
	output_image = cv2.merge((cloth_mask, mask_trash, mask_hair))
	output_resized = cv2.resize(output_image, TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)
	
	return output_resized


def process_image(image_path: Path, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> np.ndarray | None:
	"""Process a single image through the filtering pipeline.
	
	Args:
		image_path: Path to the input image.
		lower_hsv: Lower HSV bound for cloth color detection.
		upper_hsv: Upper HSV bound for cloth color detection.
		
	Returns:
		Processed image array, or None if the image cannot be read.
	"""
	bgr = cv2.imread(str(image_path))
	if bgr is None:
		print(f"Warning: Could not read image {image_path}. Skipping.")
		return None

	roi_bgr = crop_to_cloth_region(bgr, lower_hsv, upper_hsv, margin_ratio=0.06)
	h, w = roi_bgr.shape[:2]

	gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
	filtered = enhance_and_filter(gray)

	if OUTPUT_MODE == 'multichannel':
		cloth_mask = cv2.bitwise_not(filtered)
		contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		hair_contours, trash_contours = classify_contours_by_shape(contours, SHAPE_PARAMS)
		out = create_multichannel_output(h, w, cloth_mask, hair_contours, trash_contours)
	else:
		out = cv2.resize(filtered, TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)
	
	return out


def find_first_image(root: Path) -> Path | None:
	"""Find the first image file in any class directory.
	
	Args:
		root: Root directory containing class folders.
		
	Returns:
		Path to the first image found, or None if no images exist.
	"""
	exts = {'.png', '.jpg', '.jpeg', '.bmp'}
	classes = list_classes(root)
	for cls in classes:
		cls_dir = root / cls
		for p in sorted(cls_dir.iterdir()):
			if p.suffix.lower() in exts:
				return p
	return None


def estimate_hsv_range_from_sample(sample_img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
	"""Estimate cloth color HSV range from a sample image.
	
	Uses the central region of the image to estimate the cloth color.
	
	Args:
		sample_img_path: Path to a sample image containing the cloth.
		
	Returns:
		Tuple of (lower_bound, upper_bound) HSV arrays.
		
	Raises:
		FileNotFoundError: If the sample image cannot be read.
	"""
	bgr = cv2.imread(str(sample_img_path))
	if bgr is None:
		raise FileNotFoundError(f"Could not read sample image to estimate cloth color: {sample_img_path}")
	H, W = bgr.shape[:2]
	x0, x1 = int(W * 0.25), int(W * 0.75)
	y0, y1 = int(H * 0.25), int(H * 0.75)
	center = bgr[y0:y1, x0:x1]
	hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
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
	"""Main preprocessing pipeline for dataset filtering."""
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


"""Batch autoboxing over all filtered images.

This module expands the original single-image boxing logic so it can iterate
over the entire `filtering/processed` tree produced by `apply_filters.py`.

Output directory structure mirrors the input class subfolders:

boxing/
  boxed/<class>/          -> annotated images with drawn boxes
  crops/<class>/          -> individual cropped ROIs (per detection)

Example file naming:
  input: filtering/processed/hair/frame_00012_filtered.png
  annotated: boxing/boxed/hair/frame_00012_filtered_boxed.jpg
  crops: boxing/crops/hair/frame_00012_filtered_0.png, ...

Notes:
  - Size filtering keeps areas in (20, 5000) pixels; tune if needed.
  - Debug image writing can be enabled per-image.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def _autobox(img: np.ndarray,
             min_area: int = 20,
             max_area: int = 5000,
             dark_thresh: int = 100,
             light_thresh: int = 160,
             morph_kernel_size: int = 3):
    """Perform autoboxing on a pre-filtered (flat-field) image.

    Returns (annotated_image, list_of_crops)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold darker (hair) and lighter (crumb) features relative to flat background ~128
    _, mask_dark = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    _, mask_light = cv2.threshold(gray, light_thresh, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(mask_dark, mask_light)

    # Morphological cleanup
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = img.copy()
    crops = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crops.append(img[y:y + h, x:x + w])

    return annotated, crops

def autobox_candidates(filtered_image_path: str,
                       write_debug: bool = False,
                       debug_path: str = "autoboxing_debug.jpg"):
    """Backward-compatible single-image entry point.
    Returns list of cropped ROIs.
    """
    img = cv2.imread(filtered_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {filtered_image_path}")
    annotated, crops = _autobox(img)
    if write_debug:
        cv2.imwrite(debug_path, annotated)
    return crops

def process_all_filtered(input_processed_dir: str = 'filtering/processed',
                         output_root: str = 'boxing',
                         annotated_dir_name: str = 'boxed',
                         crops_dir_name: str = 'crops'):
    """Process every filtered image produced by apply_filters.

    Creates two parallel directory trees under `output_root` mirroring
    class/category subfolders from `input_processed_dir`.
    """
    if not os.path.isdir(input_processed_dir):
        raise FileNotFoundError(f"Input processed directory not found: {input_processed_dir}")

    annotated_root = os.path.join(output_root, annotated_dir_name)
    crops_root = os.path.join(output_root, crops_dir_name)
    os.makedirs(annotated_root, exist_ok=True)
    os.makedirs(crops_root, exist_ok=True)

    subfolders = [f for f in os.listdir(input_processed_dir) if os.path.isdir(os.path.join(input_processed_dir, f))]

    total_images = 0
    total_detections = 0

    for folder in subfolders:
        in_sub = os.path.join(input_processed_dir, folder)
        ann_sub = os.path.join(annotated_root, folder)
        crop_sub = os.path.join(crops_root, folder)
        os.makedirs(ann_sub, exist_ok=True)
        os.makedirs(crop_sub, exist_ok=True)

        image_files = [f for f in os.listdir(in_sub) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = sorted(image_files)

        for image_file in tqdm(image_files, desc=f"Autoboxing {folder}"):
            image_path = os.path.join(in_sub, image_file)
            img = cv2.imread(image_path)
            if img is None:
                continue
            total_images += 1
            annotated, crops = _autobox(img)
            total_detections += len(crops)

            base, ext = os.path.splitext(image_file)
            annotated_name = f"{base}_boxed.jpg"  # always jpg for annotated
            annotated_path = os.path.join(ann_sub, annotated_name)
            cv2.imwrite(annotated_path, annotated)

            for i, crop in enumerate(crops):
                crop_name = f"{base}_{i}.png"
                crop_path = os.path.join(crop_sub, crop_name)
                cv2.imwrite(crop_path, crop)

    print(f"Processed {total_images} images across {len(subfolders)} folders.")
    print(f"Total detections (crops saved): {total_detections}")

if __name__ == "__main__":
    # Default batch run
    process_all_filtered()
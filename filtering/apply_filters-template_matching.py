import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

def remove_large_noise(tophat_image, threshold_value=15, max_area=80):
    """
    Filters out large bright objects (like logos) from the tophat layer.
    
    Args:
        tophat_image: The result of the morphological TopHat (grayscale).
        threshold_value: Minimum brightness to consider a pixel as a 'feature'.
        max_area: Maximum number of pixels allowed for a crumb. 
                  Anything bigger (like a letter in a logo) is removed.
    """
    # 1. Create a binary mask of the features
    _, binary = cv2.threshold(tophat_image, threshold_value, 255, cv2.THRESH_BINARY)

    # 2. Find contours (connected blobs)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Create a mask to draw ONLY the small items (crumbs)
    clean_mask = np.zeros_like(tophat_image)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # IF the blob is smaller than max_area, we keep it.
        # IF it is larger (like a logo letter), we ignore it.
        if 0 < area < max_area:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # 4. Apply the clean mask back to the original tophat
    # This keeps the gradient/intensity of the crumbs but removes the logos
    result = cv2.bitwise_and(tophat_image, tophat_image, mask=clean_mask)
    
    return result

def morphological_contrast_enhancement(image, kernel_size=19, crumb_boost=4.0, hair_boost=4.0, shadow_gamma=0.6):
    """
    Enhances hair (dark) and crumbs (light) by extracting them and placing them 
    on a neutral gray background, while filtering out logos.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Pre-processing: Bilateral Filter
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. Gamma Correction (Lift Shadows)
    norm_img = smoothed.astype(np.float32) / 255.0
    lifted = np.power(norm_img, shadow_gamma)
    lifted_uint8 = (lifted * 255).astype(np.uint8)

    # 4. Define Morphological Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 5. White Top-Hat (Extracts Light Crumbs AND Logos)
    white_tophat = cv2.morphologyEx(lifted_uint8, cv2.MORPH_TOPHAT, kernel)

    # --- NEW STEP: FILTER LOGOS ---
    # We filter the white_tophat to remove large structures (logos) 
    # before adding them to the final image.
    # You may need to tune 'max_area' depending on how big your crumbs are.
    white_tophat_clean = remove_large_noise(white_tophat, threshold_value=10, max_area=60)
    # ------------------------------

    # 6. Black Top-Hat (Extracts Dark Hair)
    black_tophat = cv2.morphologyEx(lifted_uint8, cv2.MORPH_BLACKHAT, kernel)

    # 7. Recombination / Fusion
    flat_background = np.full_like(lifted_uint8, 128, dtype=np.float32)
    
    # Add the CLEANED crumbs
    result = flat_background + (white_tophat_clean.astype(np.float32) * crumb_boost)
    
    # Subtract the hair
    result -= (black_tophat.astype(np.float32) * hair_boost)

    # 8. Final Contrast Stretch
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    final = clahe.apply(result)

    return final

def process_images(input_dir, output_dir, num_images_per_folder=1000):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    subfolders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # Handle case where there are no subfolders (flat directory)
    if not subfolders:
        subfolders = ['.']

    for folder in subfolders:
        input_folder_path = os.path.join(input_dir, folder)
        output_folder_path = os.path.join(output_dir, folder)

        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        image_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = sorted(image_files)[:num_images_per_folder]

        print(f"Processing folder: {folder}")
        for image_file in tqdm(image_files, desc=f"Processing {folder}"):
            image_path = os.path.join(input_folder_path, image_file)
            image = cv2.imread(image_path)

            if image is None:
                continue

            filtered_image = morphological_contrast_enhancement(
                image, 
                kernel_size=19,
                crumb_boost=5.0, 
                hair_boost=5.0,    
                shadow_gamma=0.6   
            )

            output_path = os.path.join(output_folder_path, f"{os.path.splitext(image_file)[0]}_filtered.png")
            cv2.imwrite(output_path, filtered_image)

if __name__ == "__main__":
    input_dataset_dir = 'dataset/frames'
    output_processed_dir = 'filtering/processed'
    
    if not os.path.exists(input_dataset_dir):
        print(f"Error: Input directory '{input_dataset_dir}' not found.")
    else:
        process_images(input_dataset_dir, output_processed_dir, num_images_per_folder=300)
        print("Image processing complete.")
import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

def morphological_contrast_enhancement(image, kernel_size=19, crumb_boost=4.0, hair_boost=4.0, shadow_gamma=0.6):
    """
    Enhances hair (dark) and crumbs (light) by extracting them and placing them 
    on a neutral gray background to eliminate large shadows.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Pre-processing: Bilateral Filter
    # Smooths noise but keeps edge sharpness
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. Gamma Correction (Lift Shadows)
    # We stretch the darks so the hair in the shadow has enough local contrast to be detected.
    norm_img = smoothed.astype(np.float32) / 255.0
    lifted = np.power(norm_img, shadow_gamma)
    lifted_uint8 = (lifted * 255).astype(np.uint8)

    # 4. Define Morphological Kernel
    # Kernel size determines the "cutoff" size. 
    # Details smaller than this are kept. Shadows larger than this are ignored.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 5. White Top-Hat (Extracts Light Crumbs)
    # Operation: Image - Opening(Image)
    white_tophat = cv2.morphologyEx(lifted_uint8, cv2.MORPH_TOPHAT, kernel)

    # 6. Black Top-Hat (Extracts Dark Hair)
    # Operation: Closing(Image) - Image
    black_tophat = cv2.morphologyEx(lifted_uint8, cv2.MORPH_BLACKHAT, kernel)

    # 7. Recombination / Fusion (FLAT FIELD METHOD)
    # Instead of adding details back to the original image (which has the shadow),
    # we add them to a flat gray canvas (128).
    
    # Create a mid-gray background
    flat_background = np.full_like(lifted_uint8, 128, dtype=np.float32)
    
    # Add the crumbs (make them brighter than gray)
    result = flat_background + (white_tophat.astype(np.float32) * crumb_boost)
    
    # Subtract the hair (make them darker than gray)
    result -= (black_tophat.astype(np.float32) * hair_boost)

    # 8. Final Contrast Stretch
    # Clip to valid range to avoid noise/artifacts
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # CLAHE: Increases local contrast on the final result
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    final = clahe.apply(result)

    return final

def process_images(input_dir, output_dir, num_images_per_folder=1000):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    subfolders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

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

            # Apply the Morphological Pipeline with UPDATED parameters
            filtered_image = morphological_contrast_enhancement(
                image, 
                kernel_size=19,    # Larger kernel helps ignore the gradient of the hand shadow
                crumb_boost=4.0,   # High boost because we are working on flat gray
                hair_boost=4.0,    # High boost for hair visibility
                shadow_gamma=0.6   # Lifts shadows enough for the math to work
            )

            output_path = os.path.join(output_folder_path, f"{os.path.splitext(image_file)[0]}_filtered.png")
            cv2.imwrite(output_path, filtered_image)

if __name__ == "__main__":
    input_dataset_dir = 'dataset/frames'
    output_processed_dir = 'filtering/processed'
    
    # Ensure input directory exists to prevent errors if testing
    if not os.path.exists(input_dataset_dir):
        print(f"Error: Input directory '{input_dataset_dir}' not found.")
    else:
        process_images(input_dataset_dir, output_processed_dir, num_images_per_folder=300)
        print("Image processing complete.")
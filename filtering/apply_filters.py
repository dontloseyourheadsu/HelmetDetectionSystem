import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil

def morphological_contrast_enhancement(image, kernel_size=9, crumb_boost=1.5, hair_boost=1.5, shadow_gamma=0.6):
    """
    Enhances hair (dark) and crumbs (light) while suppressing background texture.
    
    Args:
        image: Input BGR image.
        kernel_size: Size of the structure element. Should be slightly larger than the width of a hair/crumb.
        crumb_boost: Multiplier to make crumbs brighter.
        hair_boost: Multiplier to make hair darker.
        shadow_gamma: Gamma < 1.0 lifts shadows to make dark hair visible in dark areas.
    """
    # 1. Convert to Grayscale (Color isn't helping much here)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Pre-processing: Bilateral Filter
    # This smooths the fabric texture (noise) but KEEPS the sharp edges of hair.
    # Gaussian blur would blur the hair too; Bilateral does not.
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. Gamma Correction (Lift Shadows)
    # We stretch the darks so the hair in the shadow isn't crushed to pure black.
    # We apply this to a float copy for calculation.
    norm_img = smoothed.astype(np.float32) / 255.0
    lifted = np.power(norm_img, shadow_gamma)
    lifted_uint8 = (lifted * 255).astype(np.uint8)

    # 4. Define Morphological Kernel
    # Ellipse is usually better than Rect for organic shapes like hair
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 5. White Top-Hat (Extracts Light Crumbs)
    # Operation: Image - Opening(Image)
    # Removes the background, leaves only things brighter/smaller than the kernel.
    white_tophat = cv2.morphologyEx(lifted_uint8, cv2.MORPH_TOPHAT, kernel)

    # 6. Black Top-Hat (Extracts Dark Hair)
    # Operation: Closing(Image) - Image
    # Removes the background, leaves only things darker/smaller than the kernel.
    black_tophat = cv2.morphologyEx(lifted_uint8, cv2.MORPH_BLACKHAT, kernel)

    # 7. Recombination / Fusion
    # Start with the shadow-lifted base
    result = lifted_uint8.astype(np.float32)
    
    # Add the crumbs (make them brighter)
    result += (white_tophat.astype(np.float32) * crumb_boost)
    
    # Subtract the hair (make them darker)
    result -= (black_tophat.astype(np.float32) * hair_boost)

    # 8. Final Contrast Stretch (Optional but recommended)
    # Clip to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # CLAHE only at the very end to normalize local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    final = clahe.apply(result)

    return final

def process_images(input_dir, output_dir, num_images_per_folder=300):
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

            # Apply the Morphological Pipeline
            filtered_image = morphological_contrast_enhancement(
                image, 
                kernel_size=13,    # Adjust based on hair thickness (bigger = catches thicker hairs)
                crumb_boost=2.0,   # How much to highlight crumbs
                hair_boost=2.5,    # How much to darken hair
                shadow_gamma=0.5   # Lower value = brighter shadows
            )

            output_path = os.path.join(output_folder_path, f"{os.path.splitext(image_file)[0]}_filtered.png")
            cv2.imwrite(output_path, filtered_image)

if __name__ == "__main__":
    input_dataset_dir = 'dataset/frames'
    output_processed_dir = 'filtering/processed'
    process_images(input_dataset_dir, output_processed_dir, num_images_per_folder=300)
    print("Image processing complete.")
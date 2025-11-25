"""Roboflow Dataset Uploader.

This script uploads processed images to a Roboflow project.
"""

import os
from pathlib import Path
from roboflow import Roboflow
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Roboflow object with your API key
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables")

rf = Roboflow(api_key=api_key)

# Specify the project for upload
workspaceId = 'feister'
projectId = 'sterilefieldmicrocontaminantdete-ljevr'
project = rf.workspace(workspaceId).project(projectId)

# Path to processed images
PROCESSED_DIR = Path(__file__).parent / "processed"
EXTENSIONS = {".jpg", ".jpeg", ".png"}

def upload_images() -> None:
    """Upload images from the processed directory to Roboflow."""
    if not PROCESSED_DIR.exists():
        print(f"Error: Directory {PROCESSED_DIR} does not exist.")
        return

    print(f"Scanning for images in {PROCESSED_DIR}...")

    # Iterate over subdirectories (classes)
    for class_dir in PROCESSED_DIR.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"Found class folder: {class_name}")
            
            # Find all images in the class directory
            images = [p for p in class_dir.glob("*") if p.suffix.lower() in EXTENSIONS]
            
            if not images:
                print(f"No images found in {class_name}")
                continue

            print(f"Uploading {len(images)} images for class '{class_name}'...")
            
            for img_path in tqdm(images, desc=f"Uploading {class_name}"):
                try:
                    # Upload image
                    # We use the folder name as a tag so Roboflow knows the class
                    project.upload(
                        image_path=str(img_path),
                        batch_name="processed_upload_v1",
                        tag_names=[class_name],
                        num_retry_uploads=3
                    )
                except Exception as e:
                    print(f"Failed to upload {img_path.name}: {e}")

if __name__ == "__main__":
    upload_images()
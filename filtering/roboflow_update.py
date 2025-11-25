import os
from pathlib import Path
from roboflow import Roboflow
from tqdm import tqdm

# Initialize the Roboflow object with your API key
rf = Roboflow(api_key="4n1r1hgpHwXxLsq2yrE9")

# Retrieve your current workspace and project name
# print(rf.workspace())

# Specify the project for upload
workspaceId = 'feister'
projectId = 'sterilefieldmicrocontaminantdete-ljevr'
project = rf.workspace(workspaceId).project(projectId)

# Path to processed images
PROCESSED_DIR = Path(__file__).parent / "processed"
EXTENSIONS = {".jpg", ".jpeg", ".png"}

def upload_images():
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
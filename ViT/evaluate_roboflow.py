"""Roboflow Workflow Evaluation.

This script evaluates the Roboflow workflow on the last N images of each class.
It calculates the average count of "White dots" and "Thin black lines" detected.
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from tqdm import tqdm
import statistics

# Load environment variables
load_dotenv()

api_key = os.getenv("ROBOFLOW_INFERENCE_API_KEY")
if not api_key:
    # Try ROBOFLOW_API_KEY as fallback
    api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    print("Error: ROBOFLOW_INFERENCE_API_KEY or ROBOFLOW_API_KEY not found.")
    exit(1)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# Configuration
WORKSPACE_NAME = "dontloseyourheadsu"
WORKFLOW_ID = "find-white-dots-and-thin-black-lines"
CLASSES = ["clean", "hair", "trash", "trash-hair"]
BASE_DIR = Path(__file__).parent.parent / "filtering" / "processed"
LIMIT = 100

results_summary = {}

print(f"Starting evaluation on last {LIMIT} images per class...")

for class_name in CLASSES:
    class_dir = BASE_DIR / class_name
    # Get all images, sort them to get the 'last' ones consistently
    images = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")))
    
    if not images:
        print(f"No images found for {class_name}")
        continue
        
    # Take last N images
    test_images = images[-LIMIT:] if len(images) > LIMIT else images
    print(f"Processing {len(test_images)} images for class '{class_name}'...")
    
    white_dots_counts = []
    black_lines_counts = []
    
    for img_path in tqdm(test_images):
        try:
            # Run workflow
            result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": str(img_path)},
                use_cache=True
            )
            
            # Parse results
            predictions = []
            if isinstance(result, list):
                result = result[0]
            
            # Look for prediction lists in the result dictionary
            for key, value in result.items():
                if isinstance(value, dict) and 'predictions' in value:
                     predictions.extend(value['predictions'])
                elif isinstance(value, list):
                     # Check if list contains prediction objects (dicts with 'class')
                     if value and isinstance(value[0], dict) and 'class' in value[0]:
                         predictions.extend(value)

            wd_count = 0
            bl_count = 0
            
            for pred in predictions:
                if 'class' in pred:
                    label = pred['class']
                    if "White dots" in label or "white dots" in label:
                        wd_count += 1
                    elif "thin black lines" in label or "black lines" in label:
                        bl_count += 1
            
            white_dots_counts.append(wd_count)
            black_lines_counts.append(bl_count)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # Calculate stats
    avg_wd = statistics.mean(white_dots_counts) if white_dots_counts else 0
    avg_bl = statistics.mean(black_lines_counts) if black_lines_counts else 0
    
    results_summary[class_name] = {
        "avg_white_dots": avg_wd,
        "avg_black_lines": avg_bl
    }

print("\n--- Roboflow SAM Evaluation Results ---")
for cls, stats in results_summary.items():
    print(f"Class: {cls}")
    print(f"  Avg White Dots: {stats['avg_white_dots']:.2f}")
    print(f"  Avg Black Lines: {stats['avg_black_lines']:.2f}")

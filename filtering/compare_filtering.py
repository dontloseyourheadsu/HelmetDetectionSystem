import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

class ImageComparator:
    """
    An application to compare original and processed images side-by-side,
    with video-like playback controls.
    """
    def __init__(self, processed_dir: Path, raw_dir: Path, cls: str, fps: int = 10):
        self.processed_cls_dir = processed_dir / cls
        self.raw_cls_dir = raw_dir / cls
        self.fps = fps
        self.window_name = f"Comparison: {cls.upper()}"

        if not self.processed_cls_dir.exists():
            raise FileNotFoundError(f"Processed class directory not found: {self.processed_cls_dir}")
        if not self.raw_cls_dir.exists():
            raise FileNotFoundError(f"Raw class directory not found: {self.raw_cls_dir}")

        self.processed_files = sorted([p for p in self.processed_cls_dir.iterdir() if p.suffix == '.png'])
        if not self.processed_files:
            raise ValueError(f"No processed images found in {self.processed_cls_dir}")

        self.current_index = 0
        self.paused = True
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _get_corresponding_raw_path(self, processed_path: Path) -> Path | None:
        """Find the original raw image path (jpg, jpeg) for a processed png."""
        stem = processed_path.stem
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            raw_path = self.raw_cls_dir / (stem + ext)
            if raw_path.exists():
                return raw_path
        return None

    def _create_overlay(self, raw_img: np.ndarray, processed_img: np.ndarray) -> np.ndarray:
        """Create a visual overlay of the processed mask on the raw image."""
        h, w = raw_img.shape[:2]
        
        # Resize processed mask to match raw image dimensions for accurate overlay
        processed_resized = cv2.resize(processed_img, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Extract channels: Blue is cloth, Green is trash, Red is hair
        cloth_mask, trash_mask, hair_mask = cv2.split(processed_resized)

        # Create colored overlays
        hair_overlay = np.zeros_like(raw_img)
        hair_overlay[hair_mask > 0] = (0, 0, 255)  # Red for hair

        trash_overlay = np.zeros_like(raw_img)
        trash_overlay[trash_mask > 0] = (0, 255, 0)  # Green for trash

        # Combine overlays and blend with the original image
        combined_overlay = cv2.add(hair_overlay, trash_overlay)
        
        # Use addWeighted for a transparent overlay effect
        overlayed_image = cv2.addWeighted(raw_img, 1.0, combined_overlay, 0.7, 0)
        
        return overlayed_image

    def run(self):
        """Main application loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while True:
            if self.current_index < 0:
                self.current_index = 0
            if self.current_index >= len(self.processed_files):
                self.current_index = len(self.processed_files) - 1

            processed_path = self.processed_files[self.current_index]
            raw_path = self._get_corresponding_raw_path(processed_path)

            # Load images
            processed_img = cv2.imread(str(processed_path), cv2.IMREAD_UNCHANGED)
            
            if raw_path and raw_path.exists():
                raw_img = cv2.imread(str(raw_path))
            else:
                # Create a black image if raw is not found
                raw_img = np.zeros((512, 512, 3), dtype=np.uint8)
                cv2.putText(raw_img, "Raw Not Found", (50, 250), self.font, 1, (255, 255, 255), 2)

            if processed_img is None:
                # Create a black image if processed is not found
                processed_img = np.zeros_like(raw_img)
                cv2.putText(processed_img, "Processed Not Found", (50, 250), self.font, 1, (255, 255, 255), 2)

            # Create the side-by-side view
            h, w, _ = raw_img.shape
            display_h = 720
            display_w = int(w * (display_h / h))
            
            raw_resized = cv2.resize(raw_img, (display_w, display_h))
            
            # Create the overlay view
            overlay_view = self._create_overlay(raw_img, processed_img)
            overlay_resized = cv2.resize(overlay_view, (display_w, display_h))

            # Add text labels
            cv2.putText(raw_resized, "Original", (10, 30), self.font, 1, (0, 255, 255), 2)
            cv2.putText(overlay_resized, "Processed Overlay (Hair=Red, Trash=Green)", (10, 30), self.font, 1, (0, 255, 255), 2)
            
            # Add frame info
            frame_info = f"Frame: {self.current_index + 1}/{len(self.processed_files)}"
            cv2.putText(raw_resized, frame_info, (10, display_h - 20), self.font, 0.8, (255, 255, 255), 2)
            if self.paused:
                cv2.putText(raw_resized, "PAUSED", (display_w - 150, display_h - 20), self.font, 0.8, (0, 0, 255), 2)


            # Combine views
            combined_view = np.hstack((raw_resized, overlay_resized))
            cv2.imshow(self.window_name, combined_view)

            # Handle keyboard input
            delay = int(1000 / self.fps) if not self.paused else 0
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
            elif key == ord('d') or key == 83:  # Right arrow
                self.current_index += 1
            elif key == ord('a') or key == 81:  # Left arrow
                self.current_index -= 1

            if not self.paused:
                self.current_index += 1

        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Compare raw and processed images.")
    script_dir = Path(__file__).resolve().parent
    
    parser.add_argument('--processed_dir', type=Path, default=(script_dir / 'processed').resolve(),
                        help="Directory containing the processed images.")
    parser.add_argument('--raw_dir', type=Path, default=(script_dir / '../dataset/frames').resolve(),
                        help="Directory containing the raw images.")
    parser.add_argument('--cls', type=str, required=True,
                        help="The class subfolder to compare (e.g., 'hair', 'clean').")
    parser.add_argument('--fps', type=int, default=10,
                        help="Frames per second for video playback.")

    args = parser.parse_args()

    try:
        comparator = ImageComparator(args.processed_dir, args.raw_dir, args.cls, args.fps)
        comparator.run()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

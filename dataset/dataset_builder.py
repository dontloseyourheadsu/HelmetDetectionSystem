"""
Dataset builder: download videos from Google Drive and extract frames.

What it does
- Uses DATASET_LINK (Google Drive folder) to download into dataset/download/.
- Finds all videos in that folder.
- Extracts a fixed number of frames per video (default 1000), evenly spaced over the full duration.
- Output goes to dataset/frames/<video-name-slug>/frame_XXXXX.png.

Run
	python dataset/dataset_builder.py [--frames-per-video 1000] [--force-download]

Requires
	pip install -r requirements.txt
"""

from __future__ import annotations

import math
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

import cv2
import gdown
import numpy as np
DATASET_LINK = "https://drive.google.com/drive/folders/1gLRc8noJhQjpkD0F6hnvUO92qi5YrbXH?usp=drive_link"

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "dataset" / "download"
FRAMES_DIR = ROOT / "dataset" / "frames"

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
CLASS_KEYS = ["sterile", "hair", "trash", "both"]
TARGET_IMAGES_PER_CLASS = 1000


def slugify(stem: str) -> str:
	s = stem.lower().strip().replace(" ", "-").replace("_", "-")
	s = "".join(ch for ch in s if ch.isalnum() or ch == "-")
	return s or "video"


def download_dataset(force: bool = False) -> None:
	"""Download the Drive folder unless videos are already present locally.

	Set force=True or pass --force-download to re-download regardless.
	"""
	DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

	existing = find_videos(DOWNLOAD_DIR)
	if existing and not force:
		print(
			f"Found {len(existing)} local video(s) in {DOWNLOAD_DIR}; skipping download. "
			f"Use --force-download to re-download."
		)
		return

	print(f"Downloading dataset into: {DOWNLOAD_DIR}")
	try:
		gdown.download_folder(DATASET_LINK, output=str(DOWNLOAD_DIR), quiet=False, use_cookies=False)
	except Exception:
		# Best-effort fallback
		gdown.download(DATASET_LINK, output=str(DOWNLOAD_DIR), quiet=False)


def find_videos(base: Path) -> List[Path]:
	vids: List[Path] = []
	for p in base.rglob("*"):
		if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
			vids.append(p)
	vids.sort()
	return vids


def classify_by_filename(videos: List[Path]) -> Dict[str, List[Path]]:
	groups: Dict[str, List[Path]] = {k: [] for k in CLASS_KEYS}
	groups["unknown"] = []
	for v in videos:
		name = v.stem.lower()
		for k in CLASS_KEYS:
			if k in name:
				groups[k].append(v)
				break
		else:
			groups["unknown"].append(v)
	return groups


def video_meta(path: Path) -> Tuple[int, float, float]:
	cap = cv2.VideoCapture(str(path))
	if not cap.isOpened():
		return 0, 0.0, 0.0
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
	duration = (total / fps) if fps > 0 else 0.0
	cap.release()
	return total, fps, duration


def compute_sampling_plan_per_video(videos: List[Path], frames_per_video: int) -> Dict[Path, int]:
	plan: Dict[Path, int] = {}
	for v in videos:
		n_frames, fps, dur = video_meta(v)
		plan[v] = min(max(1, frames_per_video), n_frames if n_frames > 0 else frames_per_video)
	return plan


def extract_frames(video_path: Path, target_count: int, out_dir: Path) -> int:
	out_dir.mkdir(parents=True, exist_ok=True)
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened() or target_count <= 0:
		return 0
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	if total <= 0:
		cap.release()
		return 0
	# Determine exact indices to sample, evenly spaced across the video
	desired = min(target_count, total)
	indices = np.linspace(0, total - 1, num=desired, dtype=np.int64)

	saved = 0
	for fidx in indices:
		cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
		ret, frame = cap.read()
		if not ret or frame is None:
			continue
		out_path = out_dir / f"frame_{saved:05d}.png"
		cv2.imwrite(str(out_path), frame)
		saved += 1
	cap.release()
	return saved


def main() -> None:
	parser = argparse.ArgumentParser(description="Download dataset and extract frames")
	parser.add_argument(
		"--force-download",
		action="store_true",
		help="Re-download from Google Drive even if local videos already exist",
	)
	parser.add_argument(
		"--frames-per-video",
		type=int,
		default=1000,
		help="Target number of frames to extract per video (default: 1000)",
	)
	args = parser.parse_args()

	print("==> Step 1/3: Downloading dataset from Drive…")
	download_dataset(force=args.force_download)

	print("==> Step 2/3: Scanning videos…")
	videos = find_videos(DOWNLOAD_DIR)
	if not videos:
		print("No videos found under dataset/download/.")
		return
	# Still classify for logging, but plan is per-video
	_ = classify_by_filename(videos)
	plan = compute_sampling_plan_per_video(videos, args.frames_per_video)

	print("Sampling summary (frames per video):")
	for v in videos:
		print(f"  {v.name}: {plan[v]} frames planned")

	print("==> Step 3/3: Extracting frames…")
	FRAMES_DIR.mkdir(parents=True, exist_ok=True)
	total_saved = 0
	for v in videos:
		out = FRAMES_DIR / slugify(v.stem)
		saved = extract_frames(v, plan.get(v, 0), out)
		print(f"{v.name}: planned {plan.get(v, 0)}, saved {saved} -> {out}")
		total_saved += saved
	print(f"Done. Total frames extracted: {total_saved}")


if __name__ == "__main__":
	main()


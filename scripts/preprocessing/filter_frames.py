"""
Epiglottis-based frame filtering for patient privacy and other.

Runs the trained YOLO model on each video, finds the first frame where the
epiglottis is detected, and keeps only frames from (detection - N) onward.
All kept frames are copied to dataset/filtered/.

Usage:
    python filter_frames.py
    python filter_frames.py --video_id 250120_LAU-0003
    python filter_frames.py --all
    python filter_frames.py --buffer 5 --conf 0.3
"""

import argparse
import shutil
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    PROJECT_ROOT, DATASET_DIR, MODEL_PATH,
    CONF_THRESHOLD, CLASSES
)

FILTERED_DIR = PROJECT_ROOT / "dataset" / "filtered"
LOG_DIR      = PROJECT_ROOT / "dataset" / "filter_logs"
BUFFER       = 5   # frames to keep before first epiglottis detection

def keep_best_per_class(result):
    """Keep only the highest confidence detection per class."""
    if result.boxes is None or len(result.boxes) == 0:
        return result
    best = {}
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if cls_id not in best or conf > best[cls_id][1]:
            best[cls_id] = (i, conf)
    keep = [v[0] for v in best.values()]
    result.boxes = result.boxes[keep]
    if result.keypoints is not None:
        result.keypoints = result.keypoints[keep]
    return result

def find_first_anatomy(model, frames: list, conf: float, class_names: list) -> int:
    """
    Return index of first frame where epiglottis OR glottis is detected.
    Tube excluded due to misclassification risk.
    """
    target_classes = {
        i for i, name in enumerate(class_names)
        if name in ("epiglottis", "glottis")
    }

    if not target_classes:
        print("  ⚠ Neither epiglottis nor glottis found in model classes")
        return -1

    for i, frame_path in enumerate(frames):
        results = model(str(frame_path), conf=conf, verbose=False)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) in target_classes:
                    return i

    return -1


def filter_video(video_id: str, model, class_names: list, conf: float, buffer: int):
    frames_dir   = DATASET_DIR / video_id
    filtered_dir = FILTERED_DIR / video_id
    log          = {"video_id": video_id, "status": None, "first_detection": None,
                    "frames_total": 0, "frames_kept": 0, "frames_discarded": 0}

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print(f"  No frames found in {frames_dir}")
        log["status"] = "no_frames"
        return log

    log["frames_total"] = len(frames)
    print(f"  [{video_id}] {len(frames)} frames — scanning for anatomy...")

    first_idx = find_first_anatomy(model, frames, conf, class_names)

    if first_idx == -1:
        print(f"  [{video_id}] ⚠ Anatomy not detected — keeping all frames")
        keep_from = 0
        log["status"] = "no_detection_keep_all"
    else:
        keep_from = max(0, first_idx - buffer)
        print(f"  [{video_id}] ✓ Anatomy first detected at frame {first_idx} ({frames[first_idx].name})")
        print(f"  [{video_id}]   Keeping from frame {keep_from} ({frames[keep_from].name}) onward")
        log["first_detection"] = frames[first_idx].name
        log["status"] = "filtered"

    kept_frames     = frames[keep_from:]
    discarded_frames = frames[:keep_from]

    log["frames_kept"]      = len(kept_frames)
    log["frames_discarded"] = len(discarded_frames)

    # Copy kept frames to filtered/
    filtered_dir.mkdir(parents=True, exist_ok=True)
    for fp in kept_frames:
        shutil.copy2(str(fp), str(filtered_dir / fp.name))

    print(f"  [{video_id}] Kept {len(kept_frames)}/{len(frames)} frames → {filtered_dir}")
    return log


def main():
    p = argparse.ArgumentParser(description="Filter frames by epiglottis detection")
    p.add_argument("--video_id", default=None, help="Single video ID")
    p.add_argument("--all",      action="store_true", help="Process all videos")
    p.add_argument("--buffer",   type=int,   default=BUFFER,
                   help=f"Frames to keep before first detection (default: {BUFFER})")
    p.add_argument("--conf",     type=float, default=CONF_THRESHOLD,
                   help=f"Detection confidence threshold (default: {CONF_THRESHOLD})")
    p.add_argument("--model",    default=str(MODEL_PATH), help="Path to YOLO weights")
    args = p.parse_args()

    from ultralytics import YOLO
    print(f"Loading model from {args.model}...")
    model       = YOLO(args.model)
    class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
    print(f"Classes: {class_names}")
    print(f"Buffer:  {args.buffer} frames before first epiglottis detection\n")

    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        video_dirs = [v for v in DATASET_DIR.iterdir() if v.is_dir()]
        print(f"Processing {len(video_dirs)} videos...\n")
    elif args.video_id:
        video_dirs = [DATASET_DIR / args.video_id]
    else:
        print("Provide --video_id or --all")
        p.print_help()
        return

    all_logs = []
    for vd in video_dirs:
        if not vd.is_dir():
            print(f"  Not found: {vd}")
            continue
        log = filter_video(vd.name, model, class_names, args.conf, args.buffer)
        all_logs.append(log)

    # Save log
    log_path = LOG_DIR / "filter_log.json"
    with open(log_path, "w") as f:
        json.dump(all_logs, f, indent=2)

    # Summary
    total_kept      = sum(l["frames_kept"]      for l in all_logs)
    total_discarded = sum(l["frames_discarded"]  for l in all_logs)
    total           = total_kept + total_discarded

    print(f"\n{'='*50}")
    print(f"  Filtering Summary")
    print(f"{'='*50}")
    print(f"  Videos processed:  {len(all_logs)}")
    print(f"  Frames total:      {total}")
    print(f"  Frames kept:       {total_kept}  ({total_kept/total*100:.1f}%)" if total else "  Frames kept: 0")
    print(f"  Frames discarded:  {total_discarded}")
    print(f"  Filtered frames →  {FILTERED_DIR}")
    print(f"  Log saved →        {log_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
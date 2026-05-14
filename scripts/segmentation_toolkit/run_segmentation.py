"""
Usage:
    # Run on one video
    python run_segmentation.py --video_id 250120_LAU-0003

    # Run on all videos
    python run_segmentation.py --all

    # Run on one video, only first N frames (for testing)
    python run_segmentation.py --video_id 250120_LAU-0003 --max_frames 20

    # Run on cluster (no display)
    python run_segmentation.py --all --headless
"""

import os
import shutil
import argparse
import subprocess
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATASET_DIR, CHECKPOINT_DIR

TOOLKIT_DIR  = CHECKPOINT_DIR / "Laryngoscopic-Image-Segmentation-Toolkit" / "Toolkit"
DATA_DIR     = CHECKPOINT_DIR / "Laryngoscopic-Image-Segmentation-Toolkit" / "data"
OUTPUT_DIR   = TOOLKIT_DIR / "output"
FRAMES_DIR   = DATASET_DIR / "frames"


def verify_setup():
    checkpoints = TOOLKIT_DIR / "checkpoints"
    required = [
        checkpoints / "best_model.dict",
        checkpoints / "yolov5_model.pt",
        checkpoints / "sam_vit_h_4b8939.pth",
        TOOLKIT_DIR / "models" / "yolov5" / "hubconf.py",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing files — run setup_toolkit.py first:")
        for m in missing:
            print(f"  ✗ {m}")
        sys.exit(1)
    print("✓ Setup verified — all weights present")


def process_video(video_id: str, max_frames: int = None, gpu: str = "0", frame_filter: str = None):
    
    frames_dir = FRAMES_DIR / video_id
    if not frames_dir.exists():
        print(f"  Frames not found: {frames_dir}")
        return 0

    frame_files = sorted(frames_dir.glob("*.jpg"))

    # ← filter FIRST, before anything else
    if frame_filter:
        frame_files = [f for f in frame_files if f.name == frame_filter]
        if not frame_files:
            print(f"  Frame not found: {frame_filter}")
            return 0

    if max_frames:
        frame_files = frame_files[:max_frames]

    if not frame_files:
        print(f"  No frames found in {frames_dir}")
        return 0

    # Create video-specific output dir
    video_output = OUTPUT_DIR / video_id
    video_output.mkdir(parents=True, exist_ok=True)

    print(f"\n[{video_id}] Processing {len(frame_files)} frames...")

    # Copy frames to toolkit data dir
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fp in frame_files:
        shutil.copy2(fp, DATA_DIR / fp.name)

    # Run toolkit on each frame
    success = 0
    for i, fp in enumerate(frame_files):
        cmd = (
            f"{sys.executable} main.py "
            f"--filename {fp.name} "
            f"--rootdir {TOOLKIT_DIR} "
            f"--gpu {gpu}"
        )
        result = subprocess.run(
            cmd, shell=True, cwd=TOOLKIT_DIR,
            capture_output=True, text=True
        )

        if result.returncode == 0:
            # Move output mask to video-specific folder
            mask_src = OUTPUT_DIR / f"{fp.stem}_mask.png"
            mask_dst = video_output / f"{fp.stem}_mask.png"
            if mask_src.exists():
                shutil.move(str(mask_src), str(mask_dst))
            success += 1
        else:
            print(f"  FAILED: {fp.name}")
            if result.stderr:
                print(f"  {result.stderr[-200:]}")

        if (i + 1) % 10 == 0 or (i + 1) == len(frame_files):
            print(f"  [{i+1}/{len(frame_files)}] {success} succeeded")

    # Clean up data dir
    for fp in frame_files:
        data_fp = DATA_DIR / fp.name
        if data_fp.exists():
            data_fp.unlink()

    print(f"  Done: {success}/{len(frame_files)} masks saved to {video_output}")
    return success


def main():
    
    p = argparse.ArgumentParser(description="Run glottis segmentation on frames")
    p.add_argument("--video_id",   help="Single video ID e.g. 250120_LAU-0003")
    p.add_argument("--all",        action="store_true", help="Process all videos")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Max frames per video (for testing)")
    p.add_argument("--gpu",        default="0",
                   help="GPU index. Use -1 for CPU (Mac)")
    p.add_argument("--frame", help="Single frame filename e.g. 000001.jpg")
    args = p.parse_args()

    verify_setup()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        video_dirs = sorted(FRAMES_DIR.iterdir())
        if not video_dirs:
            print(f"No videos found in {FRAMES_DIR}")
            return
        print(f"Processing {len(video_dirs)} videos...")
        total = 0
        for vd in video_dirs:
            if vd.is_dir():
                total += process_video(vd.name, args.max_frames, args.gpu)
        print(f"\nTotal masks generated: {total}")

    elif args.video_id:
        process_video(args.video_id, args.max_frames, args.gpu, frame_filter=args.frame)

    else:
        print("Provide --video_id or --all")
        p.print_help()


if __name__ == "__main__":
    main()
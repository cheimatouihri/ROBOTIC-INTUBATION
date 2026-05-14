"""
Script to visualize random segmentation samples in a grid format.

"""

import argparse
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATASET_DIR, RESULTS_DIR , PROJECT_ROOT

FRAMES_DIR   = DATASET_DIR / "frames"
OUTPUT_DIR   = RESULTS_DIR / "grid_segmentation_samples"


def pick_random_frames(n: int):
    """Pick n random (video_id, frame) pairs."""
    video_dirs = [v for v in FRAMES_DIR.iterdir() if v.is_dir()]
    if not video_dirs:
        print(f"No videos found in {FRAMES_DIR}")
        sys.exit(1)

    samples = []
    attempts = 0
    while len(samples) < n and attempts < n * 10:
        video = random.choice(video_dirs)
        frames = sorted(video.glob("*.jpg"))
        if frames:
            frame = random.choice(frames)
            samples.append((video.name, frame.stem))
        attempts += 1

    if len(samples) < n:
        print(f"Warning: only found {len(samples)} frames across all videos")
    return samples


def run_segmentation(video_id: str, frame: str, gpu: str):
    print(f"  Segmenting {video_id} / {frame}.jpg ...")
    result = subprocess.run(
        [sys.executable, "run_segmentation.py",
         "--video_id", video_id,
         "--frame", f"{frame}.jpg",
         "--gpu", gpu],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-300:]}")
        return False
    return True


def build_overlay(original, mask):
    if original.shape[:2] != mask.shape[:2]:
        from skimage.transform import resize
        mask = resize(mask, original.shape[:2], anti_aliasing=True)

    overlay = np.zeros((*original.shape[:2], 4), dtype=np.float32)
    mask_f  = mask[:, :, :3] if mask.ndim == 3 else mask

    unet_pixels = (mask_f[:,:,0] > 0.5) & (mask_f[:,:,1] < 0.3)
    sam_pixels  = (mask_f[:,:,1] > 0.5) & (mask_f[:,:,0] < 0.3)

    overlay[unet_pixels] = [1, 0, 0, 0.5]
    overlay[sam_pixels]  = [0, 1, 0, 0.5]
    return overlay


def visualize_grid(samples):
    """Show all samples in a grid: each row = original | mask | overlay."""
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle("Random frame segmentation samples", fontsize=14, y=1.01)

    legend = [
        mpatches.Patch(color=(1, 0, 0), label="UNet — glottis"),
        mpatches.Patch(color=(0, 1, 0), label="SAM — glottis"),
        mpatches.Patch(color=(0, 0, 0), label="Background"),
    ]

    for row, (video_id, frame) in enumerate(samples):
        original_path = FRAMES_DIR / video_id / f"{frame}.jpg"
        mask_path     = OUTPUT_DIR / video_id / f"{frame}_mask.png"

        if not original_path.exists() or not mask_path.exists():
            for col in range(3):
                axes[row][col].set_visible(False)
            continue

        original = plt.imread(str(original_path))
        mask     = plt.imread(str(mask_path))
        overlay  = build_overlay(original, mask)

        axes[row][0].imshow(original)
        axes[row][0].set_title(f"{video_id}\nframe {frame}", fontsize=8)
        axes[row][0].axis("off")

        axes[row][1].imshow(mask)
        axes[row][1].set_title("Mask", fontsize=8)
        axes[row][1].axis("off")

        axes[row][2].imshow(original)
        axes[row][2].imshow(overlay)
        axes[row][2].set_title("Overlay", fontsize=8)
        axes[row][2].axis("off")
        axes[row][2].legend(handles=legend, loc="lower right",
                            fontsize=7, framealpha=0.8, facecolor="white")

    plt.tight_layout()
    out_path = PROJECT_ROOT / "random_sample_results.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nGrid saved: {out_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n",   type=int, default=5, help="Number of random frames")
    p.add_argument("--gpu", default="-1",        help="GPU index (-1 for CPU/MPS)")
    args = p.parse_args()

    samples = pick_random_frames(args.n)
    print(f"Selected {len(samples)} random frames:")
    for video_id, frame in samples:
        print(f"  {video_id} / {frame}.jpg")

    print("\nRunning segmentation...")
    succeeded = []
    for video_id, frame in samples:
        if run_segmentation(video_id, frame, args.gpu):
            succeeded.append((video_id, frame))

    print(f"\n{len(succeeded)}/{len(samples)} frames segmented successfully")

    if succeeded:
        visualize_grid(succeeded)


if __name__ == "__main__":
    main()
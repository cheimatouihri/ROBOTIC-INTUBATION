"""
Usage:
    python visualize_mask.py --video_id 250120_LAU-0003 --frame 000066
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
FRAMES_DIR   = PROJECT_ROOT / "dataset" / "frames"
OUTPUT_DIR   = PROJECT_ROOT / "Laryngoscopic-Image-Segmentation-Toolkit" / "Toolkit" / "output"


def visualize(video_id: str, frame: str):
    original_path = FRAMES_DIR / video_id / f"{frame}.jpg"
    mask_path     = OUTPUT_DIR / video_id / f"{frame}_mask.png"

    if not original_path.exists():
        print(f"Original not found: {original_path}")
        return
    if not mask_path.exists():
        print(f"Mask not found: {mask_path}")
        return

    original = plt.imread(str(original_path))
    mask     = plt.imread(str(mask_path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{video_id} — frame {frame}", fontsize=13)

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask)
    axes[1].set_title("Segmentation mask")
    axes[1].axis("off")

    # Legend
    legend = [
        mpatches.Patch(color=(1, 0, 0), label="UNet — glottis"),
        mpatches.Patch(color=(0, 1, 0), label="SAM — glottis"),
        mpatches.Patch(color=(0, 0, 0), label="Background"),
    ]
    axes[1].legend(handles=legend, loc="lower right", fontsize=9,
                   framealpha=0.8, facecolor="white")

    plt.tight_layout()

    out_path = OUTPUT_DIR / video_id / f"{frame}_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", required=True, help="e.g. 250120_LAU-0003")
    p.add_argument("--frame",    required=True, help="e.g. 000066 (no extension)")
    args = p.parse_args()
    visualize(args.video_id, args.frame)


if __name__ == "__main__":
    main()
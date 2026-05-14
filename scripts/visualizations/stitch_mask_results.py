"""
stitch_mask_results.py — Stitch original frames with segmentation mask overlays

Usage:
    # Create overview grid for one video
    python stitch_mask_results.py --video_id 250120_LAU-0003 --mode grid

    # Create overlay video
    python stitch_mask_results.py --video_id 250120_LAU-0003 --mode video

    # Create side by side images
    python stitch_mask_results.py --video_id 250120_LAU-0003 --mode sidebyside

    # Run on all videos
    python stitch_mask_results.py --all --mode video
"""

import cv2
import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import CHECKPOINT_DIR, DATASET_DIR, RESULTS_DIR

FRAMES_DIR = DATASET_DIR / "frames"
MASKS_DIR  = CHECKPOINT_DIR / "Laryngoscopic-Image-Segmentation-Toolkit" / "Toolkit" / "output"
STITCH_DIR = RESULTS_DIR / "stitch_output"

# Overlay colors (BGR)
MASK_COLOR   = (0, 255, 0)     # green overlay for glottis/vocal folds
OVERLAY_ALPHA = 0.4            # transparency of overlay

def is_glottis_detected(mask_bgr: np.ndarray) -> bool:
    """
    Detect glottis presence from the toolkit output mask.
    Red pixels = glottis (U-Net), Green pixels = vocal folds (SAM).
    """
    if mask_bgr is None:
        return False
    red_pixels = (
        (mask_bgr[:, :, 2].astype(int) > 150) &
        (mask_bgr[:, :, 1].astype(int) < 100) &
        (mask_bgr[:, :, 0].astype(int) < 100)
    )
    return int(red_pixels.sum()) > 200


def apply_overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply colored overlay on frame where mask is non-zero."""
    out = frame_bgr.copy()

    # Handle different mask types
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    # Normalize mask to 0-255
    if mask_gray.max() <= 1.0:
        mask_gray = (mask_gray * 255).astype(np.uint8)

    # Resize mask to match frame if needed
    if mask_gray.shape[:2] != frame_bgr.shape[:2]:
        mask_gray = cv2.resize(mask_gray, (frame_bgr.shape[1], frame_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

    # Apply colored overlay
    binary = mask_gray > 127
    overlay = out.copy()
    overlay[binary] = (
        np.array(MASK_COLOR) * OVERLAY_ALPHA +
        overlay[binary] * (1 - OVERLAY_ALPHA)
    ).astype(np.uint8)

    # Draw contour
    contours, _ = cv2.findContours(
        mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, MASK_COLOR, 2)

    return overlay


def add_glottis_indicator(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Add a status indicator bar on the right side of the image.
    Green pulsing dot = glottis detected
    Red dot = no glottis
    Like a Twitch live indicator.
    """
    h, w   = img.shape[:2]
    bar_w  = 40
    out    = np.zeros((h, w + bar_w, 3), dtype=np.uint8)
    out[:, :w] = img

    # Check if mask has meaningful content
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    if mask_gray.max() <= 1.0:
        mask_gray = (mask_gray * 255).astype(np.uint8)

    detected      = mask_gray.sum() > 500   # threshold — at least 500 white pixels
    dot_color     = (0, 220, 0) if detected else (0, 0, 220)   # green or red
    label         = "GLOTTIS" if detected else "NO GLOTTIS"
    label_color   = (0, 220, 0) if detected else (0, 0, 220)

    # Draw indicator bar background
    out[:, w:] = (20, 20, 20)

    # Draw dot
    cx = w + bar_w // 2
    cy = h // 2
    cv2.circle(out, (cx, cy), 10, dot_color, -1)
    cv2.circle(out, (cx, cy), 13, dot_color, 2)   # outer ring

    # Draw vertical text label
    # Write rotated text using a temporary image
    text_img = np.zeros((80, h, 3), dtype=np.uint8)
    cv2.putText(text_img, label, (5, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1, cv2.LINE_AA)
    text_rotated = cv2.rotate(text_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Place rotated text in bar
    th, tw = text_rotated.shape[:2]
    y_start = max(0, cy - th // 2)
    y_end   = min(h, y_start + th)
    x_start = w
    x_end   = min(w + bar_w, x_start + tw)
    actual_h = y_end - y_start
    actual_w = x_end - x_start
    out[y_start:y_end, x_start:x_end] = text_rotated[:actual_h, :actual_w]

    return out, detected


def make_side_by_side(frame_bgr: np.ndarray, overlay: np.ndarray,
                      frame_name: str, detected: bool) -> np.ndarray:
    """Create side-by-side comparison with glottis status text inside the frame."""
    h, w = frame_bgr.shape[:2]

    left  = frame_bgr.copy()
    right = overlay.copy()

    cv2.putText(left,  "Original",  (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(right, "Segmented", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Status text bottom of LEFT frame
    status  = "GLOTTIS DETECTED" if detected else "GLOTTIS NOT VISIBLE"
    color   = (0, 255, 0) if detected else (0, 80, 255)
    dot_col = (0, 255, 0) if detected else (0, 0, 255)

    cv2.rectangle(left, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.circle(left,   (15, h - 17), 7, dot_col, -1)
    cv2.putText(left, status, (28, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Frame name bottom of right frame
    cv2.rectangle(right, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.putText(right, frame_name, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    return np.hstack([left, right])


def mode_sidebyside(video_id: str, output_dir: Path):
    """Save one side-by-side image per frame."""
    frames_dir = FRAMES_DIR / video_id
    masks_dir  = MASKS_DIR  / video_id
    out_dir    = output_dir / video_id / "sidebyside"
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(masks_dir.glob("*_mask.png"))
    if not mask_files:
        print(f"  No masks found in {masks_dir}")
        return 0

    saved = 0
    for mf in mask_files:
        frame_name = mf.stem.replace("_mask", "") + ".jpg"
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            continue

        frame = cv2.imread(str(frame_path))
        mask  = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue

        overlay   = apply_overlay(frame, mask)
        combined  = make_side_by_side(frame, overlay, frame_name)

        out_path  = out_dir / f"{mf.stem}_compare.jpg"
        cv2.imwrite(str(out_path), combined)
        saved += 1

    print(f"  [{video_id}] Saved {saved} side-by-side images → {out_dir}")
    return saved


def mode_video(video_id: str, output_dir: Path, fps: float = 5.0):
    """Create a video showing original + overlay side by side."""
    frames_dir = FRAMES_DIR / video_id
    masks_dir  = MASKS_DIR  / video_id
    out_dir    = output_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(masks_dir.glob("*_mask.png"))
    if not mask_files:
        print(f"  No masks found in {masks_dir}")
        return

    # Get frame size from first valid frame
    first_frame = None
    for mf in mask_files:
        frame_name = mf.stem.replace("_mask", "") + ".jpg"
        frame_path = frames_dir / frame_name
        if frame_path.exists():
            first_frame = cv2.imread(str(frame_path))
            break

    if first_frame is None:
        print(f"  No matching frames found")
        return

    h, w     = first_frame.shape[:2]
    out_path = out_dir / f"{video_id}_segmentation.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w * 2, h)
    )

    written = 0
    for mf in mask_files:
        frame_name = mf.stem.replace("_mask", "") + ".jpg"
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            continue

        frame    = cv2.imread(str(frame_path))
        mask_bgr = cv2.imread(str(mf))           # load as colour to detect red
        mask     = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue

        detected = is_glottis_detected(mask_bgr)
        overlay  = apply_overlay(frame, mask)
        combined = make_side_by_side(frame, overlay, frame_name, detected)
        writer.write(combined)
        written += 1

    writer.release()
    print(f"  [{video_id}] Video saved → {out_path}  ({written} frames)")


def mode_grid(video_id: str, output_dir: Path,
              cols: int = 4, every_n: int = 5):
    """
    Create a grid image showing every Nth frame with overlay.
    Good for a quick overview of segmentation quality.
    """
    frames_dir = FRAMES_DIR / video_id
    masks_dir  = MASKS_DIR  / video_id
    out_dir    = output_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(masks_dir.glob("*_mask.png"))[::every_n]
    if not mask_files:
        print(f"  No masks found in {masks_dir}")
        return

    cells = []
    for mf in mask_files:
        frame_name = mf.stem.replace("_mask", "") + ".jpg"
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            continue

        frame = cv2.imread(str(frame_path))
        mask  = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue

        overlay = apply_overlay(frame, mask)

        # Add frame name label
        cv2.putText(overlay, frame_name, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cells.append(overlay)

    if not cells:
        print(f"  No valid cells to display")
        return

    # Resize all cells to same size
    cell_h, cell_w = 200, 150
    cells_resized = [cv2.resize(c, (cell_w, cell_h)) for c in cells]

    # Build grid
    rows = (len(cells_resized) + cols - 1) // cols
    grid_rows = []
    for r in range(rows):
        row_cells = cells_resized[r*cols:(r+1)*cols]
        # Pad last row if needed
        while len(row_cells) < cols:
            row_cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        grid_rows.append(np.hstack(row_cells))
    grid = np.vstack(grid_rows)

    # Add title
    title_bar = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, f"{video_id} — Segmentation Overview (every {every_n} frames)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    grid = np.vstack([title_bar, grid])

    out_path = out_dir / f"{video_id}_grid.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  [{video_id}] Grid saved → {out_path}  ({len(cells)} frames shown)")


def print_stats(video_id: str):
    """Print segmentation coverage stats for a video."""
    frames_dir = FRAMES_DIR / video_id
    masks_dir  = MASKS_DIR  / video_id

    total_frames = len(list(frames_dir.glob("*.jpg")))
    total_masks  = len(list(masks_dir.glob("*_mask.png"))) if masks_dir.exists() else 0
    coverage     = total_masks / total_frames * 100 if total_frames > 0 else 0

    print(f"  [{video_id}]  frames={total_frames}  masks={total_masks}  "
          f"coverage={coverage:.1f}%")


def main():
    p = argparse.ArgumentParser(description="Stitch segmentation results for inspection")
    p.add_argument("--video_id", help="Single video ID")
    p.add_argument("--all",      action="store_true", help="Process all videos")
    p.add_argument("--mode",     choices=["grid", "video", "sidebyside"],
                   default="grid", help="Output mode (default: grid)")
    p.add_argument("--every_n",  type=int, default=5,
                   help="Grid mode: show every Nth frame (default: 5)")
    p.add_argument("--fps",      type=float, default=5.0,
                   help="Video mode: output fps (default: 5)")
    p.add_argument("--output_dir", default=str(STITCH_DIR))
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_ids = []
    if args.all:
        video_ids = [d.name for d in MASKS_DIR.iterdir() if d.is_dir()]
        if not video_ids:
            print(f"No segmented videos found in {MASKS_DIR}")
            return
        print(f"Processing {len(video_ids)} videos in '{args.mode}' mode\n")
    elif args.video_id:
        video_ids = [args.video_id]
    else:
        print("Provide --video_id or --all")
        p.print_help()
        return

    # Print stats first
    print("Coverage stats:")
    for vid in video_ids:
        print_stats(vid)
    print()

    # Run selected mode
    for vid in video_ids:
        if args.mode == "grid":
            mode_grid(vid, output_dir, every_n=args.every_n)
        elif args.mode == "video":
            mode_video(vid, output_dir, fps=args.fps)
        elif args.mode == "sidebyside":
            mode_sidebyside(vid, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
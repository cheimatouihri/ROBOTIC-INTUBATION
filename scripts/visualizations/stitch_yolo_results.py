"""
stitch_yolo_results.py — Stitch YOLO pose predictions into video or grid

Usage:
    # Create video for one video
    python stitch_yolo_results.py --video_id 250402_LAU-0280 --mode video

    # Create grid overview
    python stitch_yolo_results.py --video_id 250402_LAU-0280 --mode grid

    # Run on all videos
    python stitch_yolo_results.py --all --mode video
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATASET_DIR, RESULTS_DIR , MODEL_PATH , CONF_THRESHOLD

FRAMES_DIR    = DATASET_DIR / "frames"
OUTPUT_DIR    = RESULTS_DIR / "stitched_yolo"

# Colors per class (BGR)
CLASS_COLORS = {
    "glottis":    (0,   255, 255),   # cyan
    "epiglottis": (0,   165, 255),   # orange
    "tube":       (255, 255, 0  ),   # yellow
}

# Keypoint colors
KPT_COLORS = [
    (0,   255, 0  ),   # green
    (0,   0,   255),   # red
    (255, 255, 255),   # white
]

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

def draw_predictions(frame: np.ndarray, result, class_names: list) -> np.ndarray:
    """Draw bounding boxes, labels, confidence and keypoints on frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    boxes = result.boxes
    kpts  = result.keypoints

    if boxes is None or len(boxes) == 0:
        # No detections — add indicator
        cv2.rectangle(out, (0, 0), (w, 30), (0, 0, 80), -1)
        cv2.putText(out, "No detections", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        return out

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        name   = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        color  = CLASS_COLORS.get(name, (200, 200, 200))

        # Bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Keypoints
        if kpts is not None and i < len(kpts.xy):
            pts = kpts.xy[i].cpu().numpy()
            for j, (px, py) in enumerate(pts):
                if px == 0 and py == 0:
                    continue  # invisible keypoint
                kcolor = KPT_COLORS[j % len(KPT_COLORS)]
                cv2.circle(out, (int(px), int(py)), 5, kcolor, -1)
                cv2.circle(out, (int(px), int(py)), 7, (0, 0, 0), 1)  # outline

    return out


def make_side_by_side(original: np.ndarray, predicted: np.ndarray,
                      frame_name: str) -> np.ndarray:
    """Stack original and prediction side by side with labels."""
    h, w = original.shape[:2]

    left  = original.copy()
    right = predicted.copy()

    # Labels
    cv2.rectangle(left,  (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.rectangle(right, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(left,  "Original",    (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.putText(right, "YOLO Prediction", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 1)

    # Frame name at bottom
    cv2.rectangle(left,  (0, h-28), (w, h), (0, 0, 0), -1)
    cv2.putText(left, frame_name, (8, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return np.hstack([left, right])


def mode_video(video_id: str, model: YOLO, class_names: list, fps: float = 10.0):
    frames_dir = FRAMES_DIR / video_id
    frame_files = sorted(frames_dir.glob("*.jpg"))

    if not frame_files:
        print(f"  No frames found in {frames_dir}")
        return

    # Get frame size
    sample = cv2.imread(str(frame_files[0]))
    h, w   = sample.shape[:2]

    out_path = OUTPUT_DIR / f"{video_id}_yolo.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w * 2, h)
    )

    print(f"  [{video_id}] Processing {len(frame_files)} frames...")
    for i, fp in enumerate(frame_files):
        frame   = cv2.imread(str(fp))
        results = model(str(fp), conf=CONF_THRESHOLD, verbose=False)
        results = keep_best_per_class(results[0])
        pred    = draw_predictions(frame, results, class_names)
        combined = make_side_by_side(frame, pred, fp.name)
        writer.write(combined)

        if (i + 1) % 30 == 0 or (i + 1) == len(frame_files):
            print(f"    [{i+1}/{len(frame_files)}]")

    writer.release()
    print(f"  ✓ Video saved → {out_path}")


def mode_grid(video_id: str, model: YOLO, class_names: list,
              every_n: int = 10, cols: int = 4):
    frames_dir  = FRAMES_DIR / video_id
    frame_files = sorted(frames_dir.glob("*.jpg"))[::every_n]

    if not frame_files:
        print(f"  No frames found")
        return

    cell_w, cell_h = 320, 240
    cells = []

    for fp in frame_files:
        frame   = cv2.imread(str(fp))
        results = model(str(fp), conf=CONF_THRESHOLD, verbose=False)
        pred    = draw_predictions(frame, results[0], class_names)
        resized = cv2.resize(pred, (cell_w, cell_h))
        cv2.putText(resized, fp.name, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cells.append(resized)

    # Build grid
    rows = (len(cells) + cols - 1) // cols
    grid_rows = []
    for r in range(rows):
        row = cells[r*cols:(r+1)*cols]
        while len(row) < cols:
            row.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        grid_rows.append(np.hstack(row))
    grid = np.vstack(grid_rows)

    # Title
    title = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(title, f"{video_id} — YOLO Predictions (every {every_n} frames)",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    grid = np.vstack([title, grid])

    out_path = OUTPUT_DIR / f"{video_id}_grid.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  ✓ Grid saved → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Visualize YOLO pose predictions")
    p.add_argument("--video_id", help="Single video ID e.g. 250402_LAU-0280")
    p.add_argument("--all",      action="store_true", help="Process all videos")
    p.add_argument("--mode",     choices=["video", "grid"], default="video")
    p.add_argument("--fps",      type=float, default=10.0)
    p.add_argument("--every_n",  type=int,   default=10,
                   help="Grid mode: show every Nth frame")
    p.add_argument("--model",    default=str(MODEL_PATH),
                   help="Path to YOLO weights")
    p.add_argument("--conf",     type=float, default=0.3)
    args = p.parse_args()


    print(f"Loading model from {args.model}...")
    model       = YOLO(args.model)
    class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
    print(f"Classes: {class_names}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        video_dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]
        print(f"Processing {len(video_dirs)} videos...\n")
        for vd in video_dirs:
            if args.mode == "video":
                mode_video(vd.name, model, class_names, args.fps)
            else:
                mode_grid(vd.name, model, class_names, args.every_n)

    elif args.video_id:
        if args.mode == "video":
            mode_video(args.video_id, model, class_names, args.fps)
        else:
            mode_grid(args.video_id, model, class_names, args.every_n)

    else:
        print("Provide --video_id or --all")
        p.print_help()


if __name__ == "__main__":
    main()
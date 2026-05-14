"""
Test YOLOv8-pose model on random frames from different videos.
"""

import argparse
import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DATASET_DIR, RESULTS_DIR , MODEL_PATH , KPT_NAMES , ANNOTATED_VIDEOS , CONF_THRESHOLD , EXCLUDED_VIDEOS

FRAMES_DIR   = DATASET_DIR / "frames"
OUT_DIR      = RESULTS_DIR / "frame_side_by_side"

COLORS = {
    "glottis":    (0,   255, 255),
    "epiglottis": (0,   165, 255),
    "tube":       (255, 255, 0  ),
}

def keep_best_per_class(result):
    """Keep only highest confidence detection per class."""
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


def draw_predictions(frame, result, class_names):
    out   = frame.copy()
    h, w  = out.shape[:2]
    boxes = result.boxes
    kpts  = result.keypoints

    if boxes is None or len(boxes) == 0:
        cv2.putText(out, "No detections", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        return out

    for i, box in enumerate(boxes):
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        name     = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        color    = COLORS.get(name, (200, 200, 200))
        kpt_name_list = KPT_NAMES.get(name, [])

        # Bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - th - 8 if y1 - th - 8 > 0 else y2 + th + 8
        cv2.rectangle(out, (x1, label_y), (x1+tw+4, label_y+th+8), color, -1)
        cv2.putText(out, label, (x1+2, label_y+th+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Keypoints
        if kpts is not None and i < len(kpts.xy):
            pts = kpts.xy[i].cpu().numpy()
            for j, (px, py) in enumerate(pts):
                if px == 0 and py == 0:
                    continue
                cv2.circle(out, (int(px), int(py)), 6, color, -1)
                cv2.circle(out, (int(px), int(py)), 8, (0, 0, 0), 1)

    return out


def make_side_by_side(original, predicted, video_id, frame_name):
    h, w  = original.shape[:2]
    predicted = cv2.resize(predicted, (w, h))
    left  = original.copy()
    right = predicted.copy()

    for img, label, color in [
        (left,  "Original",   (200, 200, 200)),
        (right, "Prediction", (0,   255, 200)),
    ]:
        cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.rectangle(left, (0, h-28), (w, h), (0, 0, 0), -1)
    cv2.putText(left, f"{video_id} / {frame_name}", (8, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return np.hstack([left, right])


def pick_random_frames(n):
    video_dirs = [
        v for v in FRAMES_DIR.iterdir() 
        if v.is_dir() and v.name not in ANNOTATED_VIDEOS and v.name not in EXCLUDED_VIDEOS # exclude annotated
    ]
    if not video_dirs:
        print(f"No unannotated videos found in {FRAMES_DIR}")
        return []

    random.shuffle(video_dirs)
    samples = []

    for video in video_dirs:
        frames = sorted(video.glob("*.jpg"))
        if frames:
            frame = random.choice(frames)
            samples.append((video.name, frame))
        if len(samples) == n:
            break

    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_id", default=None, help="Test on specific video e.g. 250402_LAU-0280")
    p.add_argument("--n",     type=int,   default=10,  help="Number of random frames")
    p.add_argument("--conf",  type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    p.add_argument("--model", default=str(MODEL_PATH), help="Path to weights")
    args = p.parse_args()

    print(f"Loading model from {args.model}...")
    model       = YOLO(args.model)
    class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names

    if args.video_id:
        video_dir = FRAMES_DIR / args.video_id
        frames    = sorted(video_dir.glob("*.jpg"))
        samples   = [(args.video_id, f) for f in frames]
        out_video = OUT_DIR / args.video_id  # ← use args.video_id here
    else:
        samples   = pick_random_frames(args.n)
        out_video = OUT_DIR  # ← random frames go directly in OUT_DIR
        print(f"\nTesting on {len(samples)} random frames...\n")

    out_video.mkdir(parents=True, exist_ok=True)

    for video_id, frame_path in samples:
        print(f"  {video_id} / {frame_path.name}")

        results    = model(str(frame_path), conf=CONF_THRESHOLD, verbose=False)
        results[0] = keep_best_per_class(results[0])

        out_path   = out_video / frame_path.name
        results[0].save(str(out_path))

    print(f"\n✓ Results saved to {OUT_DIR}/")
    print(f"  Open with: open {OUT_DIR}/")


if __name__ == "__main__":
    main()
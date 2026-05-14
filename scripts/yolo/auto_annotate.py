"""
Auto-annotation script — runs trained YOLOv8-pose model on unannotated frames
and exports predictions in both COCO and YOLOv8 formats for human review.

Usage:
    python auto_annotate.py --n_videos 5
    python auto_annotate.py --n_videos 10 --frames_per_video 20
    python auto_annotate.py --video_id 250402_LAU-0280

To upload to Roboflow:
    mkdir roboflow_upload
    cp results/auto_annotations/yolo/images/*.jpg roboflow_upload/
    cp results/auto_annotations/auto_annotations.coco.json roboflow_upload/_annotations.coco.json
    zip -r roboflow_upload.zip roboflow_upload/
"""

import json
import random
import argparse
import cv2
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (
    PROJECT_ROOT, DATASET_DIR, MODEL_PATH, RESULTS_DIR,
    ANNOTATED_VIDEOS, CLASSES, KPT_NAMES, KPT_SHAPE, CONF_THRESHOLD
)

OUT_DIR = RESULTS_DIR / "auto_annotations"
FRAMES_DIR = DATASET_DIR / "frames"

def sample_frames(n_videos: int, frames_per_video: int) -> list:
    """Pick frames_per_video random frames from n_videos unannotated videos."""
    video_dirs = [
        v for v in FRAMES_DIR.iterdir()
        if v.is_dir() and v.name not in ANNOTATED_VIDEOS
    ]
    if not video_dirs:
        print("No unannotated videos found.")
        return []

    random.shuffle(video_dirs)
    selected = video_dirs[:n_videos]

    samples = []
    for video in selected:
        frames = sorted(video.glob("*.jpg"))
        if not frames:
            continue
        picked = random.sample(frames, min(frames_per_video, len(frames)))
        samples.extend([(video.name, f) for f in sorted(picked)])

    print(f"Selected {len(samples)} frames from {len(selected)} videos")
    return samples


def get_video_frames(video_id: str) -> list:
    """Get all frames from a specific video."""
    video_dir = FRAMES_DIR / video_id
    frames    = sorted(video_dir.glob("*.jpg"))
    return [(video_id, f) for f in frames]


def build_coco_export(predictions: list) -> dict:
    """Build a COCO JSON from model predictions."""
    categories = []
    for i, cls_name in enumerate(CLASSES):
        kpt_names = KPT_NAMES.get(cls_name, [])
        categories.append({
            "id":          i,
            "name":        cls_name,
            "supercategory": "anatomy",
            "keypoints":   kpt_names,
            "skeleton":    []
        })

    images      = []
    annotations = []
    ann_id      = 0
    img_id      = 0

    for video_id, frame_path, result, class_names in predictions:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        images.append({
            "id":        img_id,
            "file_name": f"{video_id}/{frame_path.name}",
            "width":     w,
            "height":    h,
        })

        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id >= len(class_names):
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                bw = x2 - x1
                bh = y2 - y1

                # Keypoints
                kpts_flat = []
                if result.keypoints is not None and i < len(result.keypoints.xy):
                    pts = result.keypoints.xy[i].cpu().numpy()
                    vis = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else None
                    for j, (px, py) in enumerate(pts):
                        v = 2 if (vis is not None and vis[j] > 0.5) else 1
                        if px == 0 and py == 0:
                            v = 0
                        kpts_flat.extend([float(px), float(py), v])

                annotations.append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": cls_id,
                    "bbox":        [x1, y1, bw, bh],
                    "area":        bw * bh,
                    "keypoints":   kpts_flat,
                    "num_keypoints": sum(1 for k in range(2, len(kpts_flat), 3) if kpts_flat[k] > 0),
                    "score":       conf,
                    "iscrowd":     0,
                    "segmentation": []
                })
                ann_id += 1

        img_id += 1

    return {
        "info":        {"description": "Auto-annotations", "date_created": str(datetime.now())},
        "categories":  categories,
        "images":      images,
        "annotations": annotations,
    }


def save_yolo_labels(predictions: list, out_dir: Path):
    """Save predictions as YOLOv8 pose label txt files."""
    labels_dir = out_dir / "yolo" / "labels"
    images_dir = out_dir / "yolo" / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    for video_id, frame_path, result, class_names in predictions:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = []
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                kpt_str = ""
                if result.keypoints is not None and i < len(result.keypoints.xy):
                    pts = result.keypoints.xy[i].cpu().numpy()
                    vis = result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else None
                    for j, (px, py) in enumerate(pts):
                        v = 2 if (vis is not None and vis[j] > 0.5) else 1
                        if px == 0 and py == 0:
                            v = 0
                        kpt_str += f" {px/w:.6f} {py/h:.6f} {v}"

                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}{kpt_str}")

        label_name = f"{video_id}_{frame_path.stem}.txt"
        (labels_dir / label_name).write_text("\n".join(lines))

        # Copy image
        import shutil
        shutil.copy(str(frame_path), str(images_dir / f"{video_id}_{frame_path.name}"))

    print(f"  ✓ YOLO labels → {labels_dir}")


def keep_best_per_class(result):
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


def main():
    p = argparse.ArgumentParser(description="Auto-annotate frames using trained YOLO model")
    p.add_argument("--n_videos",         type=int,   default=5)
    p.add_argument("--frames_per_video", type=int,   default=20)
    p.add_argument("--video_id",         default=None)
    p.add_argument("--conf",             type=float, default=CONF_THRESHOLD)
    p.add_argument("--model",            default=str(MODEL_PATH))
    args = p.parse_args()

    from ultralytics import YOLO
    print(f"Loading model from {args.model}...")
    model       = YOLO(args.model)
    class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names

    # Sample frames
    if args.video_id:
        samples = get_video_frames(args.video_id)
    else:
        samples = sample_frames(args.n_videos, args.frames_per_video)

    if not samples:
        return

    picked_videos = list(set(video_id for video_id, _ in samples))
    print(f"\nVideos selected:")
    for v in picked_videos:
        print(f"  {v}")
    print()

    # Run inference
    print(f"\nRunning inference on {len(samples)} frames...")
    predictions = []
    for i, (video_id, frame_path) in enumerate(samples):
        results    = model(str(frame_path), conf=args.conf, verbose=False)
        results[0] = keep_best_per_class(results[0])
        predictions.append((video_id, frame_path, results[0], class_names))
        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}]")

    # Export
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # COCO
    coco_data = build_coco_export(predictions)
    coco_path = OUT_DIR / "auto_annotations.coco.json"
    with open(coco_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"  ✓ COCO export → {coco_path}")
    print(f"    {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")

    # YOLO
    save_yolo_labels(predictions, OUT_DIR)

    print(f"\n✓ Done. Upload {coco_path} to Roboflow for human review.")


if __name__ == "__main__":
    main()
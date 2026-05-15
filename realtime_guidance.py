"""
realtime_guidance.py — Simulates real-time intubation guidance on a recorded video.

Runs YOLOv8-pose inference frame by frame, displays:
- Bounding boxes + keypoints for glottis, epiglottis, tube
- Estimated esophagus position (anatomical prior)
- Safety indicator: distance from tube tip to estimated esophagus
- Guidance vector: tube tip → glottic centroid

Usage:
    python realtime_guidance.py --video dataset/videos/250122_LAU-0011.mp4
    python realtime_guidance.py --video dataset/videos/250122_LAU-0011.mp4 --fps 10
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.resolve()))
from config import PROJECT_ROOT, MODEL_PATH, CONF_THRESHOLD


CLASS_COLORS = {
    "glottis":    (0,   255, 255),  # cyan
    "epiglottis": (0,   165, 255),  # orange
    "tube":       (255, 255, 0  ),  # yellow
    "esophagus":  (0,   0,   255),  # red
}

KPT_COLORS = [
    (0,   255, 0  ),
    (0,   0,   255),
    (255, 255, 255),
]

# Anatomical offset: esophagus estimated position relative to glottis posterior commissure
# In laryngoscopic view, esophagus is posterior, appears below glottis
# These values were chosen empirically
ESOPHAGUS_OFFSET_Y = 0.15   # fraction of frame height below glottis posterior commissure
ESOPHAGUS_OFFSET_X = 0.05   # slightly on right

# Safety thresholds (fraction of frame width)
SAFE_DISTANCE     = 0.15    # green
WARNING_DISTANCE  = 0.08    # yellow
DANGER_DISTANCE   = 0.04    # red


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


def get_keypoint(result, class_names, class_name, kpt_idx=0):
    """Get a specific keypoint (x,y) for a class. Returns None if not found."""
    if result.boxes is None:
        return None
    for i, box in enumerate(result.boxes):
        name = class_names[int(box.cls[0])]
        if name == class_name and result.keypoints is not None:
            if i < len(result.keypoints.xy):
                pts = result.keypoints.xy[i].cpu().numpy()
                if kpt_idx < len(pts):
                    px, py = pts[kpt_idx]
                    if not (px == 0 and py == 0):
                        return (int(px), int(py))
    return None


def estimate_esophagus(result, class_names, frame_h, frame_w):
    """
    Estimate esophagus position from glottis posterior commissure (kpt index 2).
    Anatomically: esophagus lies immediately posterior to glottic inlet.
    Reference: StatPearls Airway Management; Paediatric Emergencies Anatomy.
    """
    posterior_commissure = get_keypoint(result, class_names, "glottis", kpt_idx=2)
    if posterior_commissure is None:
        # Fall back to centroid if posterior commissure not detected
        posterior_commissure = get_keypoint(result, class_names, "glottis", kpt_idx=0)
    if posterior_commissure is None:
        return None

    cx, cy = posterior_commissure
    ex = int(cx + ESOPHAGUS_OFFSET_X * frame_w)
    ey = int(cy + ESOPHAGUS_OFFSET_Y * frame_h)
    return (max(0, min(frame_w-1, ex)), max(0, min(frame_h-1, ey)))


def compute_distance(p1, p2, frame_w):
    """Compute normalized distance between two points."""
    if p1 is None or p2 is None:
        return None
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist / frame_w


def safety_color(distance):
    if distance is None:
        return (150, 150, 150), "UNKNOWN"
    if distance > SAFE_DISTANCE:
        return (0, 220, 0),   "SAFE"
    elif distance > WARNING_DISTANCE:
        return (0, 165, 255), "WARNING"
    else:
        return (0, 0, 255),   "DANGER"


def draw_frame(frame, result, class_names):
    out   = frame.copy()
    h, w  = out.shape[:2]

    glottis_centroid = None
    tube_tip         = None

    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            name   = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            color  = CLASS_COLORS.get(name, (200, 200, 200))

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = y1 - th - 8 if y1 - th - 8 > 0 else y2 + th + 8
            label_x = min(x1, w - tw - 4)
            cv2.rectangle(out, (label_x, label_y), (label_x+tw+4, label_y+th+8), color, -1)
            cv2.putText(out, label, (label_x+2, label_y+th+2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            # Keypoints
            if result.keypoints is not None and i < len(result.keypoints.xy):
                pts = result.keypoints.xy[i].cpu().numpy()
                for j, (px, py) in enumerate(pts):
                    if px == 0 and py == 0:
                        continue
                    kcolor = KPT_COLORS[j % len(KPT_COLORS)]
                    cv2.circle(out, (int(px), int(py)), 5, kcolor, -1)
                    cv2.circle(out, (int(px), int(py)), 7, (0, 0, 0), 1)

                # Store glottis centroid (keypoint 0) and tube tip (keypoint 0)
                if name == "glottis" and len(pts) > 0:
                    px, py = pts[0]
                    if not (px == 0 and py == 0):
                        glottis_centroid = (int(px), int(py))
                if name == "tube" and len(pts) > 0:
                    px, py = pts[0]
                    if not (px == 0 and py == 0):
                        tube_tip = (int(px), int(py))

    # Estimated esophagus
    esophagus_pos = estimate_esophagus(result, class_names, h, w)
    if esophagus_pos:
        ex, ey = esophagus_pos
        # Draw red ⊗ marker
        cv2.circle(out, (ex, ey), 12, (0, 0, 220), 2)
        cv2.line(out, (ex-12, ey), (ex+12, ey), (0, 0, 220), 2)
        cv2.line(out, (ex, ey-12), (ex, ey+12), (0, 0, 220), 2)
        cv2.putText(out, "ESO*", (ex+14, ey+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)

    # Guidance vector: tube tip → glottis centroid
    # if tube_tip and glottis_centroid:
    #     cv2.arrowedLine(out, tube_tip, glottis_centroid,
    #                     (255, 255, 255), 2, tipLength=0.15)

    # Safety indicator
    dist  = compute_distance(tube_tip, esophagus_pos, w)
    color, status = safety_color(dist)

    # Status bar at bottom
    bar_h = 40
    cv2.rectangle(out, (0, h-bar_h), (w, h), (20, 20, 20), -1)

    # Safety dot
    cv2.circle(out, (20, h-bar_h//2), 10, color, -1)

    # Status text
    status_text = f"{status}"
    if dist is not None:
        status_text += f"  |  Esophagus dist: {dist*100:.1f}%"
    else:
        status_text += "  | "

    cv2.putText(out, status_text, (38, h-bar_h//2+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # ESO disclaimer
    cv2.putText(out, "*Estimated position", (w-180, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    return out


def run_on_frames(frames_dir: Path, model, class_names, target_fps, conf):
    """Run on a folder of frames simulating real-time."""
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    print(f"Running on {len(frames)} frames at {target_fps}fps")
    print("Press Q to quit\n")

    frame_delay = 1.0 / target_fps

    for fp in frames:
        t_start = time.time()

        frame   = cv2.imread(str(fp))
        results = model(str(fp), conf=conf, verbose=False)
        results[0] = keep_best_per_class(results[0])
        out     = draw_frame(frame, results[0], class_names)

        cv2.imshow("Robotic Intubation Guidance", out)

        # Maintain target fps
        elapsed = time.time() - t_start
        wait_ms = max(1, int((frame_delay - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def run_on_video(video_path: Path, model, class_names, target_fps, conf):
    """Run on a video file simulating real-time."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Running on video: {video_path.name} ({total} frames) at {target_fps}fps")
    print("Press Q to quit\n")

    frame_delay = 1.0 / target_fps

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Save temp frame for YOLO
        tmp = PROJECT_ROOT / ".tmp_frame.jpg"
        cv2.imwrite(str(tmp), frame)

        results = model(str(tmp), conf=conf, verbose=False)
        results[0] = keep_best_per_class(results[0])
        out = draw_frame(frame, results[0], class_names)

        cv2.imshow("Robotic Intubation Guidance", out)

        elapsed = time.time() - t_start
        wait_ms = max(1, int((frame_delay - elapsed) * 1000))
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    tmp.unlink(missing_ok=True)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Real-time intubation guidance simulation")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video",  help="Path to video file (.mp4)")
    g.add_argument("--frames", help="Path to folder of frames")
    p.add_argument("--fps",   type=float, default=8.0, help="Playback speed (default: 8)")
    # to match the inference bottleneck ~77ms per frame on cpu
    p.add_argument("--conf",  type=float, default=CONF_THRESHOLD)
    p.add_argument("--model", default=str(MODEL_PATH))
    args = p.parse_args()

    from ultralytics import YOLO
    print(f"Loading model from {args.model}...")
    model       = YOLO(args.model)
    class_names = list(model.names.values()) if isinstance(model.names, dict) else model.names
    print(f"Classes: {class_names}\n")

    if args.video:
        run_on_video(Path(args.video), model, class_names, args.fps, args.conf)
    else:
        run_on_frames(Path(args.frames), model, class_names, args.fps, args.conf)


if __name__ == "__main__":
    main()
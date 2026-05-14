"""
Visualize COCO annotations with bounding boxes, keypoints and keypoint names.
Usage:
    python visualize_annotations.py --frame 000066_jpg.rf.xxxxx.jpg
"""

import json
import cv2
import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import ANNOTATION_DIR , KPT_NAMES , CLASSES

ANNOTATION_JSON = ANNOTATION_DIR / "train" / "_annotations.coco.json"
IMAGES_DIR      = ANNOTATION_DIR / "train" / "images"
OUT_DIR         = ANNOTATION_DIR / "vizualizations"

COLORS = {
    "glottis":    (0,   255, 255),  # cyan
    "epiglottis": (0,   165, 255),  # orange
    "tube":       (255, 255, 0  ),  # yellow
}


def draw_annotation(img, ann, cat_name, color):
    # Bounding box
    x, y, w, h = [int(float(v)) for v in ann["bbox"]]
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Class label
    #(tw, th), _ = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    bw = w  # box width
    bh = h  # box height — from ann["bbox"]
    img_h, img_w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    # Try above box first, fall back to below, clamp horizontally
    if y - th - 6 >= 0:
        label_y = y - th - 6
    elif y + bh + th + 6 <= img_h:
        label_y = y + bh
    else:
        label_y = y - th - 6  # force above even if slightly clipped

    label_x = min(x, img_w - tw - 4)

    cv2.rectangle(img, (label_x, label_y), (label_x+tw+4, label_y+th+6), color, -1)
    cv2.putText(img, cat_name, (label_x+2, label_y+th+2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    # cls_idx = CLASSES.index(cat_name) if cat_name in CLASSES else 0
    # offset  = cls_idx * (th + 10)
    # label_y = y - th - 8 - offset if y - th - 8 - offset > 0 else y + h + th + 8 + offset
    # cv2.rectangle(img, (x, label_y), (x+tw+4, label_y+th+8), color, -1)
    # cv2.putText(img, cat_name, (x+2, label_y+th+2),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Keypoints
    kpts     = ann.get("keypoints", [])
    kpt_name_list = KPT_NAMES.get(cat_name, [])
    n        = len(kpts) // 3

    for i in range(n):
        kx = int(kpts[i*3])
        ky = int(kpts[i*3 + 1])
        kv = kpts[i*3 + 2]

        if kv == 0 or (kx == 0 and ky == 0):
            continue  # invisible

        # Draw keypoint dot
        cv2.circle(img, (kx, ky), 6, color, -1)
        cv2.circle(img, (kx, ky), 8, (0, 0, 0), 1)

        # Draw keypoint name
        kpt_label = kpt_name_list[i] if i < len(kpt_name_list) else f"kp{i}"
        cv2.putText(img, kpt_label, (kx + 10, ky + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return img


def visualize(image_info, annotations, cats, out_path):
    img_path = IMAGES_DIR / image_info["file_name"]
    img      = cv2.imread(str(img_path))

    if img is None:
        print(f"  Could not read: {img_path}")
        return False

    for ann in annotations:
        cat_name = cats.get(ann["category_id"], "unknown")
        if cat_name in ("nose", "RoboticIntubation"):
            continue
        color = COLORS.get(cat_name, (200, 200, 200))
        draw_annotation(img, ann, cat_name, color)

    cv2.imwrite(str(out_path), img)
    print(f"  Saved: {out_path}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame",    help="Exact filename e.g. 000066_jpg.rf.xxxxx.jpg")
    p.add_argument("--frame_id", help="Original frame number e.g. 000066")
    p.add_argument("--n",        type=int, default=5, help="Number of images (default: 5)")
    args = p.parse_args()

    with open(ANNOTATION_JSON) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    cats   = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Group annotations by image
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    OUT_DIR.mkdir(exist_ok=True)

    if args.frame:
        # Find by exact filename
        matches = [img for img in data["images"] if img["file_name"] == args.frame]
        if not matches:
            print(f"Frame not found: {args.frame}")
            return
        selected = matches

    elif args.frame_id:
        # Find by original frame number prefix
        matches = [img for img in data["images"] if img["file_name"].startswith(args.frame_id)]
        if not matches:
            print(f"No frames found starting with: {args.frame_id}")
            return
        selected = matches

    else:
        # First N images
        selected = data["images"][:args.n]

    for img_info in selected:
        anns     = anns_by_image.get(img_info["id"], [])
        out_path = OUT_DIR / img_info["file_name"]
        visualize(img_info, anns, cats, out_path)

    print(f"\nDone. Open: open {OUT_DIR}/")


if __name__ == "__main__":
    main()
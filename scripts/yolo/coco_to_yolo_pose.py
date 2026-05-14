"""
Convert COCO keypoint annotations to YOLOv8 pose format.

Output structure:
    annotation/
        train/
            images/     
            labels/    (created by this script in YOLOv8 pose format)
        valid/
            images/
            labels/
"""

import json
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import PROJECT_ROOT, ANNOTATION_DIR , CLASSES, KPT_SHAPE

def convert_split(split: str):
    split_dir = ANNOTATION_DIR / split
    if not split_dir.exists():
        return None

    # Find annotation JSON
    coco_json = None
    for name in ["_annotations.coco.json", "annotation.json", "_annotations.json"]:
        candidate = split_dir / name
        if candidate.exists():
            coco_json = candidate
            break

    if not coco_json:
        print(f"  No annotation JSON found in {split_dir}")
        return None

    print(f"  Using: {coco_json.name}")
    with open(coco_json) as f:
        data = json.load(f)

    images     = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat for cat in data["categories"]}

    # Max keypoints from actual annotations (not category metadata)
    max_kpts = max(
        (len(ann.get("keypoints", [])) // 3 for ann in data["annotations"]),
        default=0
    )

    # Class list excluding 'nose' and parent class 'RoboticIntubation'
    class_names = [
        cat["name"]
        for cat in sorted(data["categories"], key=lambda x: x["id"])
        if cat["name"] in CLASSES
    ]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"  Classes: {class_names}")
    print(f"  Max keypoints: {max_kpts}")

    # Output labels dir
    labels_dir = split_dir / "labels"
    labels_dir.mkdir(exist_ok=True)

    # Group annotations by image
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    converted = 0
    for img_id, img_info in images.items():
        img_w = img_info["width"]
        img_h = img_info["height"]
        stem  = Path(img_info["file_name"]).stem

        anns  = anns_by_image.get(img_id, [])
        lines = []

        for ann in anns:
            cat_name = categories[ann["category_id"]]["name"]
            if cat_name not in class_to_idx:
                continue

            class_idx = class_to_idx[cat_name]

            # BBox: COCO (x,y,w,h) → YOLO (cx,cy,w,h) normalized
            x, y, w, h = [float(v) for v in ann["bbox"]]
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Keypoints
            kpts  = ann.get("keypoints", [])
            n     = len(kpts) // 3
            kpt_str = ""
            for i in range(n):
                kx = kpts[i*3]     / img_w
                ky = kpts[i*3 + 1] / img_h
                kv = kpts[i*3 + 2]
                kpt_str += f" {kx:.6f} {ky:.6f} {int(kv)}"
            # Pad missing keypoints with zeros
            for _ in range(n, max_kpts):
                kpt_str += " 0.000000 0.000000 0"

            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}{kpt_str}")

        if lines:
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
            converted += 1
            
    # Move images into images/ subfolder
    images_dir = split_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for img in split_dir.glob("*.jpg"):
        img.rename(images_dir / img.name)
    for img in split_dir.glob("*.png"):
        img.rename(images_dir / img.name)
    print(f"  ✓ Images moved to {images_dir}")

    print(f"  ✓ {converted} label files written to {labels_dir}")
    return class_names, max_kpts


def update_data_yaml(class_names, max_kpts):
    out_path = ANNOTATION_DIR / "data_fixed.yaml"
    data = {
        "train": str(ANNOTATION_DIR / "train" / "images"),
        "val":   str(ANNOTATION_DIR / "valid" / "images"),
        "nc":        len(class_names),
        "names":     class_names,
        "kpt_shape": KPT_SHAPE,
    }
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  ✓ data_fixed.yaml → nc={len(class_names)}, kpt_shape=[{max_kpts}, 3]")


def main():
    print("Converting COCO → YOLOv8 Pose format...\n")

    result = None
    for split in ["train", "valid", "val"]:
        if (ANNOTATION_DIR / split).exists():
            print(f"Processing {split}/...")
            r = convert_split(split)
            if r:
                result = r

    if result:
        print("\nUpdating data_fixed.yaml...")
        update_data_yaml(*result)
        print("\n✓ Done. Now run: python train_yolo_pose.py")
    else:
        print("✗ No annotations converted — check your folder structure.")


if __name__ == "__main__":
    main()
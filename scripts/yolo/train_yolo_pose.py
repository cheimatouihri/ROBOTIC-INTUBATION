"""
Train YOLOv8-pose on laryngoscopy keypoint annotations.

Usage:
    python train_yolo_pose.py
    python train_yolo_pose.py --epochs 100 --model yolov8s-pose.pt
    python train_yolo_pose.py --resume  # resume from last checkpoint
"""

import argparse
import shutil
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import PROJECT_ROOT, ANNOTATION_DIR, CHECKPOINT_DIR , GPU , EPOCHS, BATCH, IMGSZ, BASE_MODEL, CLASSES , KPT_SHAPE, EARLY_STOPPING 

DATA_YAML = ANNOTATION_DIR / "data.yaml"
RUNS_DIR  = CHECKPOINT_DIR


def split_dataset(val_ratio: float = 0.2):
    """Split train into train/val if val folder doesn't exist."""
    import random
    import shutil

    train_images = ANNOTATION_DIR / "train" / "images"
    train_labels = ANNOTATION_DIR / "train" / "labels"
    val_images   = ANNOTATION_DIR / "valid" / "images"
    val_labels   = ANNOTATION_DIR / "valid" / "labels"

    if val_images.exists() and any(val_images.iterdir()):
        print("✓ Validation set already exists — skipping split")
        return

    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    all_images = sorted(train_images.glob("*.jpg")) + sorted(train_images.glob("*.png"))
    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_ratio))
    val_files = all_images[:n_val]

    print(f"  Splitting: {len(all_images)} total → {len(all_images)-n_val} train / {n_val} val")

    for img in val_files:
        shutil.move(str(img), str(val_images / img.name))
        label = train_labels / (img.stem + ".txt")
        if label.exists():
            shutil.move(str(label), str(val_labels / label.name))

    print(f"✓ Split complete")


def fix_data_yaml():
    """
    Use data_fixed.yaml if already created by coco_to_yolo_pose.py,
    otherwise fix data.yaml paths and kpt_shape.
    """
    fixed_yaml = ANNOTATION_DIR / "data_fixed.yaml"

    # If coco converter already created data_fixed.yaml, use it directly
    if fixed_yaml.exists():
        print(f"✓ Using existing data_fixed.yaml from COCO converter")
        # Still run split in case valid/ folder is missing
        split_dataset(val_ratio=0.2)
        return fixed_yaml

    # Otherwise fix data.yaml manually
    split_dataset(val_ratio=0.2)

    with open(DATA_YAML, "r") as f:
        data = yaml.safe_load(f)

    data["train"] = str(ANNOTATION_DIR / "train" / "images")
    data["val"]   = str(ANNOTATION_DIR / "valid" / "images")

    if not (ANNOTATION_DIR / "test").exists():
        data.pop("test", None)

    data["kpt_shape"] = KPT_SHAPE

    if "names" in data:
        data["names"] = [n for n in data["names"] if n in CLASSES]
        data["nc"]    = len(data["names"])
        print(f"  Classes: {data['names']}")

    with open(fixed_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✓ data_fixed.yaml created → {fixed_yaml}")
    return fixed_yaml


def train(epochs: int, model_name: str, resume: bool, batch: int, imgsz: int):
    from ultralytics import YOLO

    fixed_yaml = fix_data_yaml()

    if resume:
        # Find last checkpoint
        last = sorted(RUNS_DIR.glob("pose/train*/weights/last.pt"))
        if not last:
            print("No checkpoint found to resume from. Starting fresh.")
            resume = False
        else:
            ckpt = last[-1]
            print(f"Resuming from {ckpt}")
            model = YOLO(str(ckpt))
            model.train(resume=True)
            return

    print(f"\nStarting training:")
    print(f"  Model:   {model_name}")
    print(f"  Epochs:  {epochs}")
    print(f"  Batch:   {batch}")
    print(f"  ImgSize: {imgsz}")
    print(f"  Data:    {fixed_yaml}\n")

    model = YOLO(model_name)  # downloads automatically if not present

    results = model.train(
        data=str(fixed_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=str(RUNS_DIR / "pose"),
        name="train",
        exist_ok=True,
        patience=EARLY_STOPPING,    
        save=True,
        plots=True,
        device="mps" if GPU == -1 else str(GPU), 
    )

    print("\nTraining complete.")
    print(f"  Results saved to: {RUNS_DIR / 'pose' / 'train'}")


def main():
    p = argparse.ArgumentParser(description="Train YOLOv8-pose for laryngoscopy guidance")
    p.add_argument("--epochs", type=int,   default=EPOCHS,     help="Number of training epochs")
    p.add_argument("--model",  default=BASE_MODEL,             help="Base model to train from")
    p.add_argument("--batch",  type=int,   default=BATCH,      help="Batch size")
    p.add_argument("--imgsz",  type=int,   default=IMGSZ,      help="Image size")
    p.add_argument("--resume", action="store_true",            help="Resume from last checkpoint")
    args = p.parse_args()

    train(args.epochs, args.model, args.resume, args.batch, args.imgsz)


if __name__ == "__main__":
    main()
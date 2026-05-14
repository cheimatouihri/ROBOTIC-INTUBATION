"""
Plot training metrics from YOLOv8 results.csv

Usage:
    python plot_metrics.py
    python plot_metrics.py --run_dir checkpoints/pose/train
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import PROJECT_ROOT, CHECKPOINT_DIR, RESULTS_DIR

DEFAULT_RUN = CHECKPOINT_DIR / "pose" / "train"
OUT_DIR     = RESULTS_DIR / "training_plots"


def plot_metrics(run_dir: Path):
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"No results.csv found in {run_dir}")
        return

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    epochs = df["epoch"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # losses
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training & Validation Losses", fontsize=14, fontweight="bold")

    loss_pairs = [
        ("train/box_loss",  "val/box_loss",  "Box Loss"),
        ("train/pose_loss", "val/pose_loss", "Pose Loss"),
        ("train/kobj_loss", "val/kobj_loss", "Keypoint Obj Loss"),
        ("train/cls_loss",  "val/cls_loss",  "Class Loss"),
        ("train/dfl_loss",  "val/dfl_loss",  "DFL Loss"),
    ]

    for ax, (train_col, val_col, title) in zip(axes.flatten(), loss_pairs):
        if train_col in df.columns:
            ax.plot(epochs, df[train_col], label="Train", color="#4C72B0", linewidth=2)
        if val_col in df.columns:
            ax.plot(epochs, df[val_col],   label="Val",   color="#DD8452", linewidth=2, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes.flatten()[-1].set_visible(False)
    plt.tight_layout()
    loss_path = OUT_DIR / "losses.png"
    plt.savefig(str(loss_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Losses → {loss_path}")

    # detection metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Detection Metrics (Bounding Box)", fontsize=14, fontweight="bold")

    det_metrics = [
        ("metrics/precision(B)", "Precision",  "#2ca02c"),
        ("metrics/recall(B)",    "Recall",     "#d62728"),
        ("metrics/mAP50(B)",     "mAP@50",     "#9467bd"),
        ("metrics/mAP50-95(B)",  "mAP@50-95",  "#8c564b"),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), det_metrics):
        if col in df.columns:
            ax.plot(epochs, df[col], color=color, linewidth=2)
            ax.fill_between(epochs, df[col], alpha=0.1, color=color)
            best_val = df[col].max()
            best_ep  = df[col].idxmax()
            ax.axvline(best_ep, color=color, linestyle=":", alpha=0.7)
            ax.set_title(f"{title}  (best: {best_val:.3f} @ epoch {best_ep})", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    det_path = OUT_DIR / "detection_metrics.png"
    plt.savefig(str(det_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Detection metrics → {det_path}")

    # pose metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pose Estimation Metrics (Keypoints)", fontsize=14, fontweight="bold")

    pose_metrics = [
        ("metrics/precision(P)", "Precision (Pose)",  "#2ca02c"),
        ("metrics/recall(P)",    "Recall (Pose)",     "#d62728"),
        ("metrics/mAP50(P)",     "mAP@50 (Pose)",     "#9467bd"),
        ("metrics/mAP50-95(P)",  "mAP@50-95 (Pose)",  "#8c564b"),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), pose_metrics):
        if col in df.columns:
            ax.plot(epochs, df[col], color=color, linewidth=2)
            ax.fill_between(epochs, df[col], alpha=0.1, color=color)
            best_val = df[col].max()
            best_ep  = df[col].idxmax()
            ax.axvline(best_ep, color=color, linestyle=":", alpha=0.7)
            ax.set_title(f"{title}  (best: {best_val:.3f} @ epoch {best_ep})", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pose_path = OUT_DIR / "pose_metrics.png"
    plt.savefig(str(pose_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Pose metrics → {pose_path}")

    # summary
    print(f"\n{'='*45}")
    print(f"  Training Summary")
    print(f"{'='*45}")
    print(f"  Total epochs:       {len(df)}")

    for col, label in [
        ("metrics/mAP50(B)",    "Best mAP50 (box)"),
        ("metrics/mAP50-95(B)", "Best mAP50-95 (box)"),
        ("metrics/mAP50(P)",    "Best mAP50 (pose)"),
        ("metrics/mAP50-95(P)", "Best mAP50-95 (pose)"),
        ("metrics/precision(B)","Best Precision"),
        ("metrics/recall(B)",   "Best Recall"),
    ]:
        if col in df.columns:
            print(f"  {label:<25} {df[col].max():.4f}  (epoch {df[col].idxmax()})")

    print(f"{'='*45}\n")
    print(f"✓ All plots saved to {OUT_DIR}/")


def main():
    p = argparse.ArgumentParser(description="Plot YOLOv8 training metrics")
    p.add_argument("--run_dir", default=str(DEFAULT_RUN), help="Path to training run folder")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    print(f"Reading results from {run_dir}...")
    plot_metrics(run_dir)


if __name__ == "__main__":
    main()
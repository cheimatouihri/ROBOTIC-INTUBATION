"""
Main entry point for the Robotic Intubation Guidance pipeline.

Usage:
    python main.py --step preprocess
    python main.py --step segment --video_id 250120_LAU-0003
    python main.py --step segment --all
    python main.py --step convert
    python main.py --step train
    python main.py --step train --epochs 50   # override config
    python main.py --step test
    python main.py --step test --video_id 250402_LAU-0280
    python main.py --step visualize --video_id 250402_LAU-0280 --fps 3
    python main.py --step visualize --all
    python main.py --step auto_annotate
    python main.py --step auto_annotate --video_id 250402_LAU-0280
    python main.py --step auto_annotate --n_videos 
    python main.py --step plot
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SCRIPTS      = PROJECT_ROOT / "scripts"

from config import (
    EPOCHS, BATCH, IMGSZ, BASE_MODEL,
    CONF_THRESHOLD, GPU, EARLY_STOPPING,
    ANNOTATED_VIDEOS
)


def install_requirements():
    requirements = PROJECT_ROOT / "requirements.txt"
    print("Checking dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", str(requirements), "-q"
    ])
    print("✓ All dependencies satisfied\n")


def run(script: Path, flags: list = []):
    cmd = [sys.executable, str(script)] + flags
    print(f"\n▶ Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"\n✗ Step failed: {script.name}")
        sys.exit(result.returncode)


def step_preprocess(args):
    run(SCRIPTS / "preprocessing" / "preprocess_laryngoscopy.py")


def step_segment(args):
    script = SCRIPTS / "segmentation_toolkit" / "run_segmentation.py"
    flags  = ["--gpu", str(args.gpu or GPU)]
    if args.all:
        flags += ["--all"]
    elif args.video_id:
        flags += ["--video_id", args.video_id]
    if args.frame:
        flags += ["--frame", args.frame]
    if args.max_frames:
        flags += ["--max_frames", str(args.max_frames)]
    run(script, flags)


def step_convert(args):
    run(SCRIPTS / "yolo" / "coco_to_yolo_pose.py")


def step_train(args):
    script = SCRIPTS / "yolo" / "train_yolo_pose.py"
    flags  = [
        "--model",  args.model  or BASE_MODEL,
        "--epochs", str(args.epochs or EPOCHS),
        "--imgsz",  str(args.imgsz  or IMGSZ),
        "--batch",  str(args.batch  or BATCH),
    ]
    if args.resume:
        flags += ["--resume"]
    run(script, flags)


def step_test(args):
    script = SCRIPTS / "visualizations" / "test_frames.py"
    flags  = [
        "--conf", str(args.conf or CONF_THRESHOLD),
        "--n",    str(args.n),
    ]
    if args.video_id:
        flags += ["--video_id", args.video_id]
    run(script, flags)


def step_visualize(args):
    script = SCRIPTS / "visualizations" / "stitch_yolo_results.py"
    flags  = [
        "--conf", str(args.conf or CONF_THRESHOLD),
        "--fps",  str(args.fps),
        "--mode", args.mode,
    ]
    if args.all:
        flags += ["--all"]
    elif args.video_id:
        flags += ["--video_id", args.video_id]
    run(script, flags)


def step_auto_annotate(args):
    script = SCRIPTS / "yolo" / "auto_annotate.py"
    flags  = [
        "--conf",             str(args.conf or CONF_THRESHOLD),
        "--n_videos",         str(args.n_videos),
        "--frames_per_video", str(args.frames_per_video),
    ]
    if args.video_id:
        flags += ["--video_id", args.video_id]
    run(script, flags)

def step_plot(args):
    script = SCRIPTS / "visualizations" / "plot_metrics.py"
    run(script)

def main():
    install_requirements()

    p = argparse.ArgumentParser(
        description="Robotic Intubation Guidance — Pipeline Runner"
    )

    p.add_argument("--step", required=True, choices=[
        "preprocess", "segment", "convert", "train",
        "test", "visualize", "auto_annotate", "plot"
    ])

    # Shared
    p.add_argument("--video_id",   default=None)
    p.add_argument("--all",        action="store_true")
    p.add_argument("--gpu",        default=None,  help=f"GPU index (config default: {GPU})")

    # Segmentation
    p.add_argument("--frame",      default=None)
    p.add_argument("--max_frames", type=int, default=None)

    # Training — defaults come from config.py
    p.add_argument("--model",   default=None, help=f"config default: {BASE_MODEL}")
    p.add_argument("--epochs",  type=int, default=None, help=f"config default: {EPOCHS}")
    p.add_argument("--imgsz",   type=int, default=None, help=f"config default: {IMGSZ}")
    p.add_argument("--batch",   type=int, default=None, help=f"config default: {BATCH}")
    p.add_argument("--patience", type=int, default=None, help=f"config default: {EARLY_STOPPING}")
    p.add_argument("--resume",  action="store_true")

    # Inference — default from config.py
    p.add_argument("--conf",    type=float, default=None, help=f"config default: {CONF_THRESHOLD}")
    p.add_argument("--n",       type=int,   default=10)

    # Visualization
    p.add_argument("--fps",     type=float, default=10.0)
    p.add_argument("--mode",    default="video", choices=["video", "grid"])

    # Auto-annotation
    p.add_argument("--n_videos",         type=int, default=5)
    p.add_argument("--frames_per_video", type=int, default=20)

    # plot 
    p.add_argument("--plot", action="store_true")

    args = p.parse_args()

    steps = {
        "preprocess":    step_preprocess,
        "segment":       step_segment,
        "convert":       step_convert,
        "train":         step_train,
        "test":          step_test,
        "visualize":     step_visualize,
        "auto_annotate": step_auto_annotate,
    }

    steps[args.step](args)


if __name__ == "__main__":
    main()
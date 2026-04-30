import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
TOOLKIT_DIR  = PROJECT_ROOT / "Laryngoscopic-Image-Segmentation-Toolkit" / "Toolkit"
CHECKPOINTS  = TOOLKIT_DIR / "checkpoints"
MODELS_DIR   = TOOLKIT_DIR / "models"
DATA_DIR     = PROJECT_ROOT / "Laryngoscopic-Image-Segmentation-Toolkit" / "data"
OUTPUT_DIR   = TOOLKIT_DIR / "output"


def run(cmd, cwd=None):
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"WARNING: command exited with code {result.returncode}")
    return result.returncode

def step0_clone_toolkit():
    toolkit_parent = PROJECT_ROOT / "Laryngoscopic-Image-Segmentation-Toolkit"
    if toolkit_parent.exists():
        print("Done! Toolkit already cloned")
        return
    print("\nCloning toolkit")
    run("git clone https://github.com/yucongzh/Laryngoscopic-Image-Segmentation-Toolkit.git",
        cwd=PROJECT_ROOT)


def step1_clone_yolov5():
    yolo_dir = MODELS_DIR / "yolov5"
    if yolo_dir.exists():
        print("Done! YOLOv5 already cloned")
        return
    print("\nCloning YOLOv5")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run("git clone https://github.com/ultralytics/yolov5.git",
        cwd=MODELS_DIR)
    # Install YOLOv5 requirements
    run(f"{sys.executable} -m pip install -q -r {yolo_dir}/requirements.txt")


def step2_download_pretrained_weights():
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    unet_path  = CHECKPOINTS / "best_model.dict"
    yolo_path  = CHECKPOINTS / "yolov5_model.pt"

    if unet_path.exists() and yolo_path.exists():
        print("Done! U-Net and YOLOv5 weights already downloaded")
        return

    print("\n─Downloading pretrained weights from HuggingFace")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        run(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download

    if not unet_path.exists():
        print("  Downloading U-Net weights (best_model.dict)...")
        hf_hub_download(
            repo_id="yucongzh/glottis_segmentation",
            filename="best_model.dict",
            local_dir=str(CHECKPOINTS)
        )
        print(f"  Done! Saved to {unet_path}")

    if not yolo_path.exists():
        print("  Downloading YOLOv5 weights (yolov5_model.pt)...")
        hf_hub_download(
            repo_id="yucongzh/glottis_segmentation",
            filename="yolov5_model.pt",
            local_dir=str(CHECKPOINTS)
        )
        print(f"Saved to {yolo_path}")


def step3_download_sam():
    sam_path = CHECKPOINTS / "sam_vit_h_4b8939.pth"
    if sam_path.exists():
        print("SAM vit_h checkpoint already downloaded")
        return
 
    print("\nDownloading SAM vit_h checkpoint")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
 
    # using curl if wget not available 
    import platform
    if platform.system() == "Darwin":
        run(f'curl -L --progress-bar "{url}" -o "{sam_path}"')
    else:
        run(f'wget -q --show-progress -O "{sam_path}" {url}')
    print(f"Saved to {sam_path}")


def step4_install_dependencies():
    print("\nInstalling Python dependencies")
    deps = [
        "torch torchvision",
        "scikit-image",
        "matplotlib",
        "opencv-python",
        "numpy",
        "Pillow",
        "git+https://github.com/facebookresearch/segment-anything.git",
    ]
    for dep in deps:
        run(f"{sys.executable} -m pip install -q {dep}")
    print("  Done! All dependencies installed")


def step5_create_dirs():
    print("\n Creating directories")
    for d in [CHECKPOINTS, MODELS_DIR, DATA_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Done! {d}")


def print_summary():
    print("\n" + "=" * 55)
    print("  SETUP COMPLETE")
    print("=" * 55)
    print(f"  Toolkit   : {TOOLKIT_DIR}")
    print(f"  Weights   : {CHECKPOINTS}")
    print(f"  Data      : {DATA_DIR}")
    print(f"  Output    : {OUTPUT_DIR}")
    print("=" * 55)
    print("\nNext: python run_segmentation.py --video_id 250120_LAU-0003")
    print("=" * 55)

    # Verify all weights present
    print("\nWeight files:")
    for f in ["best_model.dict", "yolov5_model.pt", "sam_vit_h_4b8939.pth"]:
        path = CHECKPOINTS / f
        status = "path exists" if path.exists() else "!! MISSING"
        size   = f"{path.stat().st_size / 1e6:.0f}MB" if path.exists() else ""
        print(f"  {status} {f} {size}")


if __name__ == "__main__":
    print("Laryngoscopic Image Segmentation Toolkit — Setup")
    print("=" * 55)

    step0_clone_toolkit()
    step5_create_dirs()
    step1_clone_yolov5()
    step2_download_pretrained_weights()
    step3_download_sam()
    step4_install_dependencies()
    print_summary()
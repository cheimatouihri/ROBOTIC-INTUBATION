
``` 
PROJECT/
├── scripts/
│   ├── preprocessing/
│   ├── segmentation_toolkit/
|   ├── visualizations/
│   └── yolo/
├── checkpoints/          ← trained YOLO weights (gitignored)
│   └── runs/pose/train/weights/best.pt
├── results/     
├── annotation/           ← gitignored
├── dataset/              ← gitignored
├── realtime_guidance.py
└── README.md
```
## en train de modifier la description en dessous

## STEPS
### Clone the segmentation toolkit

```bash
git clone https://github.com/yucongzh/Laryngoscopic-Image-Segmentation-Toolkit.git
```
### Run setup script to download model weights and install dependencies automatically

```bash
python setup_toolkit.py
```

### Extracting frames 

```bash 
python preprocess_laryngoscopy.py
```
Frames are saved to `dataset/frames/<video_id>/`.

### Run glottis segmentation
 
On a single video:
```bash
python run_segmentation.py --video_id 250120_LAU-0003 --gpu -1
```
 
On a single frame (for testing):
```bash
python run_segmentation.py --video_id 250120_LAU-0003 --frame 000066.jpg --gpu -1
```

On all videos:
```bash
python run_segmentation.py --all --gpu -1
```

Masks are saved to:
```
Laryngoscopic-Image-Segmentation-Toolkit/Toolkit/output/<video_id>/<frame>_mask.png
```

### Visualize results
 
Single frame (original | mask | overlay):
```bash
python visualize_mask.py --video_id 250120_LAU-0003 --frame 000066
```
 
Random frames across random videos:
```bash
python run_random_sample.py --gpu -1
```
Output saved as `random_sample_results.png` in project root.

## Pipeline Overview
 
```
Video frames (dataset/frames/)
        ↓
preprocess_laryngoscopy.py     — extracts frames from raw videos
        ↓
run_segmentation.py            — UNet segments glottis → binary masks
        ↓
masks_to_keypoints.py          — extracts keypoints from masks
        ↓
(We might skip) 
[Label Studio]                 — human review + tube keypoint annotation
        ↓
train_yolo_pose.py             — trains YOLOv8-pose on keypoints
        ↓
realtime_guidance.py           — live inference → robotic control signals
```

## Segmentation: What We Know So Far
 
We use the [Laryngoscopic Image Segmentation Toolkit](https://github.com/yucongzh/Laryngoscopic-Image-Segmentation-Toolkit), which combines two models:
 
| Model | Role | Performance on our data |
|-------|------|------------------------|
| UNet | Glottis detection | good |
| SAM (vit_h) | Vocal fold refinement | Inconsistent |
 
**Color legend in output masks:**
| Color | Meaning |
|-------|---------|
| Red | UNet — glottis |
| Green | SAM — vocal folds |
| Black | Background |
 
UNet was trained on the BAGLS dataset (59,250 laryngoscopy frames). SAM is a general-purpose model not tuned for endoscopic images — we use UNet masks as the primary output and will refine during annotation.

## Scripts Reference
| Script | Purpose | Usage |
|--------|---------|-------|
| `preprocess_laryngoscopy.py` | Extract frames from videos | `python preprocess_laryngoscopy.py` |
| `run_segmentation.py` | Run glottis segmentation | `python run_segmentation.py --video_id 250120_LAU-0003 --gpu -1` |
| `visualize_mask.py` | View original + mask + overlay | `python visualize_mask.py --video_id 250120_LAU-0003 --frame 000066` |
| `run_random_sample.py` | Segment & visualize 5 random frames | `python run_random_sample.py --gpu -1` |

---

## Current state 

### Annotation
 
1. Export frames to [Roboflow](https://roboflow.com) — project type: **Keypoint Detection**

2. Annotate with 3 keypoints per class:

| Class | Keypoints |
|-------|-----------|
| glottis | centroid, anterior commissure, posterior commissure |
| epiglottis | tip, left, right |
| tube | tip, mid, base |
 
3. Export dataset as **COCO** format → download zip → place in `annotation/`
---
 
### Training
 
**`coco_to_yolo_pose.py`** — Convert Roboflow COCO export to YOLOv8 pose format
```bash
python coco_to_yolo_pose.py
```
Run this once after downloading from Roboflow. Creates `annotation/train/labels/` and `annotation/data_fixed.yaml`.
 
**`train_yolo_pose.py`** — Train YOLOv8-pose model
```bash
# First run
python train_yolo_pose.py --model yolov8n-pose.pt --epochs 100 
```
Best weights saved to `runs/pose/train/weights/best.pt` and copied to `best.pt`.
 
 
---
 
### Evaluation & Visualization
 
**`stitch_yolo_results.py`** — Run YOLO on a full video and visualize predictions
```bash
# Side-by-side video (original vs predictions)
python stitch_yolo_results.py --video_id 250402_LAU-0280 --mode video --fps 10
# decrease fps for slower video

# Grid overview
python stitch_yolo_results.py --video_id 250402_LAU-0280 --mode grid --every_n 10
 
# All videos
python stitch_yolo_results.py --all --mode video
 
```
Output: `yolo_output/<video_id>/<video_id>_yolo.mp4`
 
**Quick single frame test:**
```bash
python -c "
from ultralytics import YOLO
model = YOLO('runs/pose/train/weights/best.pt')
results = model('dataset/frames/<video_id>/<frame>.jpg', conf=0.3)
results[0].save('test_prediction.png')
"
open test_prediction.png
```


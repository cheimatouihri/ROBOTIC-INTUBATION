# Robotic intubation – predictive airway modeling and data-driven guidance
This project focuses on the development of predictive airway analytics and morphological modeling to support robotic intubation, leveraging large datasets to improve anatomical understanding and real-time navigation during airway management.

``` 
PROJECT/
├── scripts/
│   ├── preprocessing/
│   ├── segmentation_toolkit/
|   ├── visualizations/
│   └── yolo/
├── checkpoints/                    ← trained YOLO weights (gitignored)
│   |── pose/train/weights/best.pt
|   └── Laryngoscopic-Image-Segmentation-Toolkit/Toolkit
├── results/     
├── annotation/ (from roboflow)     ← gitignored
├── dataset/                        ← gitignored
|   ├── videos                    
|   └── frames                   
├── realtime_guidance.py
└── README.md
```


## Environment Setup
 
### Create conda environment (Python 3.11 required)
```bash
conda create -n robotic-intubation python=3.11
conda activate robotic-intubation
pip install -r requirements.txt
```
 
### Dataset
```
dataset/videos/<video_id>.mp4
dataset/frames/<video_id>/*.jpg
```
 
---
 
## Configuration
 
All paths, classes, and hyperparameters are centralized in `config.py`. Edit this file before running anything:
 
```python
# Classes to detect
CLASSES = ["glottis", "epiglottis", "tube", "esophagus"]
 
# Training hyperparameters
EPOCHS   = 200
BATCH    = 8
IMGSZ    = 320
PATIENCE = 40
 
# Videos with ground truth annotations (excluded from auto-annotation & testing)
ANNOTATED_VIDEOS = ["250120_LAU-0003", ...]
 
# Videos excluded from all processing (privacy/quality issues)
EXCLUDED_VIDEOS  = []
```
 
---
 
## Running the Pipeline
 
Everything runs through `main.py`:
 
```bash
python main.py --step <step> [options]
```
 
### Available steps
 
| Step | Description | Command |
|------|-------------|---------|
| `preprocess` | Extract frames from videos | `python main.py --step preprocess` |
| `filter` | Filter frames by epiglottis/glottis detection | `python main.py --step filter --all` |
| `segment` | Run UNet+SAM segmentation (exploration) | `python main.py --step segment --video_id ID` |
| `convert` | Convert Roboflow COCO → YOLOv8 pose | `python main.py --step convert` |
| `train` | Train YOLOv8-pose model | `python main.py --step train` |
| `test` | Test model on unseen frames | `python main.py --step test` |
| `visualize` | Run model on full video | `python main.py --step visualize --video_id ID` |
| `auto_annotate` | Auto-annotate frames for Roboflow review | `python main.py --step auto_annotate` |
 
All defaults come from `config.py` and can be overridden via CLI:
```bash
python main.py --step train --epochs 50 --batch 4
python main.py --step test --video_id 250402_LAU-0280 --conf 0.3
python main.py --step visualize --video_id 250402_LAU-0280 --fps 3
```
 
---
 
## Full Pipeline
 
```
dataset/videos/
        ↓
preprocess          — extract frames → dataset/frames/
        ↓
filter              — keep only clinically relevant frames → dataset/filtered/
        ↓
[Roboflow]          — manual keypoint annotation
        ↓
convert             — COCO → YOLOv8 pose format
        ↓
train               — train YOLOv8-pose
        ↓
auto_annotate       — auto-label new frames → upload to Roboflow for review
        ↓
test / visualize    — evaluate model on unseen videos
        ↓
realtime_guidance   — live inference → robotic control signals (TODO)
```
 
---
 
## Annotation
 
Annotations are managed in [Roboflow](https://roboflow.com) — project type: **Keypoint Detection**.
 
### Keypoints per class
 
| Class | Keypoints | Notes |
|-------|-----------|-------|
| glottis | centroid, anterior, posterior | Tracheal entrance |
| epiglottis | tip, left, right | Must be lifted for safe intubation |
| tube | tip, mid, base | Tube excluded from frame filter (misclassification risk) |
| (not used) esophagus | center | tube must not enter here |
 
### Export & convert
1. Export from Roboflow as **COCO** format → place in `annotation/`
2. Run conversion:
```bash
python main.py --step convert
```
 
### Auto-annotation workflow 
(didn't work)
1. Run auto-annotate on unannotated videos:
```bash
python main.py --step auto_annotate --n_videos 5 --frames_per_video 20
```
2. Zip output for Roboflow upload:
```bash
mkdir roboflow_upload
cp results/auto_annotations/yolo/images/*.jpg roboflow_upload/
cp results/auto_annotations/auto_annotations.coco.json roboflow_upload/_annotations.coco.json
zip -r roboflow_upload.zip roboflow_upload/
```
3. Upload zip to Roboflow → review predictions → correct → re-export → retrain
---
 
## Training
 
```bash
# Train with config defaults (200 epochs, patience 40)
python main.py --step train
 
# Override specific params
python main.py --step train --epochs 100 --batch 4
 
# Resume from checkpoint
python main.py --step train --resume
```
 
Best weights saved to `checkpoints/pose/train/weights/best.pt`.
 
### Plot training metrics
```bash
python scripts/visualizations/plot_metrics.py
```
Output: `results/training_plots/`
 
---
 
## Evaluation
 
```bash
# Test on 10 random unseen frames
python main.py --step test
 
# Test on specific video (all frames)
python main.py --step test --video_id 250402_LAU-0280
 
# Full video side-by-side (original vs prediction)
python main.py --step visualize --video_id 250402_LAU-0280 --fps 3
 
# Grid overview
python main.py --step visualize --video_id 250402_LAU-0280 --mode grid
```
 
---
 
## Privacy & Ethics
 
- **Frame filtering** (`filter` step) detects first epiglottis/glottis appearance and discards preceding frames that may contain identifying patient or OR information
- **5 frames** before first anatomy detection are retained as context
- Videos with repeated camera withdrawal or OR visibility should be added to `EXCLUDED_VIDEOS` in `config.py`
- Dataset folders are gitignored 
---
 
## Key Design Decisions
 
| Decision | Reason |
|----------|--------|
| YOLOv8-pose over UNet+SAM | Single model, real-time capable, outputs keypoints directly |
| Keypoints over masks | Guidance signal only needs positions, not pixel-level segmentation |
| 3 keypoints per class | Consistent across all classes → `kpt_shape=[3,3]` |
| Roboflow for annotation | Fast keypoint annotation with COCO export |
| Epiglottis/glottis for frame filtering | Tube excluded due to misclassification risk |
| Human review for problematic videos | Automated filtering cannot handle repeated camera withdrawal |
 
---
 
## References
 
- **YOLOv8-pose:** Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* (Version 8.0.0). https://github.com/ultralytics/ultralytics
- **Laryngoscopic Segmentation Toolkit:** https://github.com/yucongzh/Laryngoscopic-Image-Segmentation-Toolkit
- **BAGLS Dataset:** Fehling, M.K. et al. (2020). Fully automatic segmentation of glottis and vocal folds in endoscopic laryngeal high-speed videos.

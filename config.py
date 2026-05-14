from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# Paths
DATASET_DIR    = PROJECT_ROOT / "dataset"
ANNOTATION_DIR = PROJECT_ROOT / "annotation"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR    = PROJECT_ROOT / "results"
MODEL_PATH     = CHECKPOINT_DIR / "pose" / "train" / "weights" / "best.pt"

# Inference
CONF_THRESHOLD = 0.5
GPU            = -1  

# Training
EPOCHS   = 200
BATCH    = 8
IMGSZ    = 320
EARLY_STOPPING = 40
BASE_MODEL    = "yolov8n-pose.pt"

# classes used in annotation 
CLASSES  = ["glottis", "epiglottis", "tube"]
KPT_NAMES = {
    "glottis":    ["centroid", "anterior", "posterior"],
    "epiglottis": ["tip", "left", "right"],
    "tube":       ["tip", "mid", "base"],
}
KPT_SHAPE = [3, 3]

# Annotated videos 
ANNOTATED_VIDEOS = [
    "250120_LAU-0003",
    "250121_LAU-0008",
    "250120_LAU-0005",
    "250122_LAU-0010",
]

# esopahgus visible 
ESOPHAGUS_VISIBLE = [
    "250121_LAU-0008",
]

# Extreme cases 
EXCLUDED_VIDEOS = [
    "250401_LAU-0272", # patient face and body clearly visible 
]


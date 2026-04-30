"""
Laryngoscopy Video Preprocessing Pipeline

Usage:
    python preprocess_laryngoscopy.py --video_dir data/videos --output_dir dataset
    python preprocess_laryngoscopy.py --video_dir data/videos --output_dir dataset --no_hw
    python preprocess_laryngoscopy.py --video_dir data/videos --output_dir dataset --hw_backend cuda
"""

import os
import cv2
import subprocess
import json
import csv
import logging
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    video_dir:  str = "./videos"
    output_dir: str = "./dataset"

    # ── Frame extraction ───────────────────────────────────────────────────
    fps: float = 5.0
    # At 30fps source → fps=5 keeps 1 in 6 frames
    # Recommended range: 3–10 fps

    # ── Quality filters ────────────────────────────────────────────────────
    blur_threshold: float = 1.5
    # Endoscopic cameras score 2–5 on sharp frames.
    # 1.5 catches only truly unusable frames.

    min_brightness: float = 25.0
    max_brightness: float = 230.0

    # ── Near-duplicate removal ─────────────────────────────────────────────
    hash_diff_threshold: int = 6
    # 0 = disabled

    # ── Output ─────────────────────────────────────────────────────────────
    image_quality: int = 92
    resize_width:  int = 640   # 0 = keep original

    # ── Hardware ───────────────────────────────────────────────────────────
    use_hw_accel: bool = True
    hw_backend: str = "videotoolbox"
    # macOS (Apple Silicon + Intel) : "videotoolbox"
    # Linux/Windows NVIDIA GPU      : "cuda"
    # Linux AMD GPU                 : "vaapi"
    # CPU fallback                  : --no_hw


def setup_logging(log_dir: Path) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = log_dir / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("laryngoscopy")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    for h in [logging.StreamHandler(), logging.FileHandler(log_path)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def mean_brightness(gray):
    return float(gray.mean())

def perceptual_hash(gray, size=8):
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()

def hash_distance(h1, h2):
    return int(np.sum(h1 != h2))


class FrameFilter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_hash = None

    def reset(self):
        self._last_hash = None

    def assess(self, frame_path: Path):
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            return False, "unreadable"
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        b = mean_brightness(gray)
        if b < self.cfg.min_brightness: return False, f"dark({b:.0f})"
        if b > self.cfg.max_brightness: return False, f"overexposed({b:.0f})"

        bl = blur_score(gray)
        if bl < self.cfg.blur_threshold: return False, f"blurry({bl:.1f})"

        if self.cfg.hash_diff_threshold > 0:
            h = perceptual_hash(gray)
            if self._last_hash is not None:
                dist = hash_distance(h, self._last_hash)
                if dist < self.cfg.hash_diff_threshold:
                    return False, f"duplicate(d={dist})"
            self._last_hash = h

        return True, "ok"


def probe_video(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-select_streams", "v:0", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        stream = json.loads(r.stdout)["streams"][0]
        num, den = map(int, stream["r_frame_rate"].split("/"))
        return {
            "width":    int(stream.get("width", 0)),
            "height":   int(stream.get("height", 0)),
            "fps":      num / den,
            "duration": float(stream.get("duration", 0)),
        }
    except Exception:
        return {}


def extract_frames(video_path: Path, out_dir: Path, cfg: Config,
                   trim_sec: float, logger) -> list:
    """
    Extract frames starting from trim_sec using ffmpeg.
    trim_sec = 0 means start from beginning.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "%06d.jpg")
    scale   = f",scale={cfg.resize_width}:-2" if cfg.resize_width > 0 else ""
    vf      = f"fps={cfg.fps}{scale}"

    # -ss before -i = fast seek (keyframe accurate enough for our use)
    seek = ["-ss", str(trim_sec)] if trim_sec > 0 else []

    def cmd(hw):
        return (
            ["ffmpeg", "-y", "-loglevel", "error"]
            + seek
            + (["-hwaccel", cfg.hw_backend] if hw else [])
            + ["-i", str(video_path)]
            + ["-vf", vf, "-q:v", "2", "-threads", "0", pattern]
        )

    try:
        subprocess.run(cmd(cfg.use_hw_accel), check=True, timeout=600)
    except subprocess.CalledProcessError:
        if cfg.use_hw_accel:
            logger.warning(f"HW accel ({cfg.hw_backend}) failed, retrying CPU...")
            subprocess.run(cmd(False), check=True, timeout=600)
        else:
            raise

    return sorted(out_dir.glob("*.jpg"))


def process_video(video_path: Path, cfg: Config, output_dir: Path,
                  trim_sec: float, logger) -> dict:
    vid_id     = video_path.stem
    frames_dir = output_dir / "frames"   / vid_id
    reject_dir = output_dir / "rejected" / vid_id

    info = probe_video(video_path)
    if info:
        logger.info(
            f"  {info['width']}x{info['height']} @ {info['fps']:.1f}fps  "
            f"duration={info['duration']:.1f}s"
        )

    if trim_sec > 0:
        logger.info(f"  Trim: starting extraction at t={trim_sec:.2f}s")
    else:
        logger.info(f"  No trim — extracting from start")

    logger.info(f"  Extracting @ {cfg.fps}fps...")
    frame_paths = extract_frames(video_path, frames_dir, cfg, trim_sec, logger)
    logger.info(f"  {len(frame_paths)} frames extracted")

    # Quality filter
    ffilter  = FrameFilter(cfg)
    accepted = []
    rejected_items = []
    rejection_counts = {}

    for fp in frame_paths:
        keep, reason = ffilter.assess(fp)
        if keep:
            accepted.append(fp)
        else:
            rejected_items.append((fp, reason))
            tag = reason.split("(")[0]
            rejection_counts[tag] = rejection_counts.get(tag, 0) + 1

    # Re-save accepted at configured JPEG quality
    for fp in accepted:
        img = cv2.imread(str(fp))
        if img is not None:
            cv2.imwrite(str(fp), img, [cv2.IMWRITE_JPEG_QUALITY, cfg.image_quality])

    # Move rejected
    if rejected_items:
        reject_dir.mkdir(parents=True, exist_ok=True)
        for fp, _ in rejected_items:
            fp.rename(reject_dir / fp.name)

    total    = len(frame_paths)
    kept     = len(accepted)
    keep_pct = kept / total * 100 if total else 0.0

    logger.info(
        f"  Kept {kept}/{total} ({keep_pct:.1f}%)  |  "
        + "  ".join(f"{k}={v}" for k, v in rejection_counts.items())
    )

    return {
        "video_id":         vid_id,
        "source":           str(video_path),
        "original_fps":     info.get("fps"),
        "duration_s":       info.get("duration"),
        "resolution":       f"{info.get('width')}x{info.get('height')}",
        "trim_sec":         trim_sec,
        "frames_extracted": total,
        "frames_kept":      kept,
        "frames_rejected":  total - kept,
        "keep_rate_pct":    round(keep_pct, 1),
        "rejection_counts": rejection_counts,
        "accepted_frames":  [str(p) for p in accepted],
    }


def run(cfg: Config):
    output_dir = Path(cfg.output_dir)
    log_dir    = output_dir / "logs"
    logger     = setup_logging(log_dir)

    video_dir = Path(cfg.video_dir)
    videos    = sorted(video_dir.glob("*.mp4"))
    if not videos:
        logger.error(f"No .mp4 files found in: {video_dir.resolve()}")
        return

    logger.info("=" * 60)
    logger.info("  Laryngoscopy Preprocessing Pipeline")
    logger.info(f"  Videos   : {len(videos)}")
    logger.info(f"  FPS      : {cfg.fps}")
    logger.info(f"  HW accel : {cfg.hw_backend if cfg.use_hw_accel else 'disabled (CPU)'}")
    logger.info(f"  Output   : {output_dir.resolve()}")
    logger.info("=" * 60)

    all_stats    = []
    all_accepted = []
    t0 = datetime.now()

    for i, vp in enumerate(videos, 1):
        logger.info(f"\n[{i}/{len(videos)}] {vp.name}")
        try:
            stats = process_video(vp, cfg, output_dir, 0.0, logger)
            all_stats.append(stats)
            all_accepted.extend(stats["accepted_frames"])
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            all_stats.append({"video_id": vp.stem, "error": str(e)})

    elapsed = (datetime.now() - t0).total_seconds()

    # manifest.csv
    manifest = output_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_path", "video_id", "label"])
        for fp in all_accepted:
            w.writerow([fp, Path(fp).parent.name, ""])

    # per_video_stats.json
    for s in all_stats:
        s.pop("accepted_frames", None)
    with open(log_dir / "per_video_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    total_extracted = sum(s.get("frames_extracted", 0) for s in all_stats)
    total_kept      = sum(s.get("frames_kept",      0) for s in all_stats)
    overall_pct     = total_kept / total_extracted * 100 if total_extracted else 0

    logger.info("\n" + "=" * 55)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 55)
    logger.info(f"  Videos processed : {len(videos)}")
    logger.info(f"  Frames extracted : {total_extracted:,}")
    logger.info(f"  Frames kept      : {total_kept:,}  ({overall_pct:.1f}%)")
    logger.info(f"  Frames rejected  : {total_extracted - total_kept:,}")
    logger.info(f"  Elapsed          : {elapsed/60:.1f} min")
    logger.info(f"  Manifest         : {manifest}")
    logger.info("=" * 55)
    logger.info("  → SAM 2: use frames/ for zero-shot segmentation")
    logger.info("  → Annotation: load manifest.csv into Label Studio / CVAT")
    logger.info("=" * 55 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Laryngoscopy preprocessing — Apple Silicon")
    p.add_argument("--video_dir",      default="./videos",    help="Folder with .mp4 files")
    p.add_argument("--output_dir",     default="./dataset",   help="Output folder")
    p.add_argument("--fps",            type=float, default=5.0)
    p.add_argument("--blur_threshold", type=float, default=1.5)
    p.add_argument("--min_brightness", type=float, default=25.0)
    p.add_argument("--max_brightness", type=float, default=230.0)
    p.add_argument("--resize",         type=int,   default=640)
    p.add_argument("--no_hw",          action="store_true")
    p.add_argument("--hw_backend",     default="videotoolbox",
                   help="videotoolbox (macOS) | cuda (NVIDIA) | vaapi (AMD)")
    p.add_argument("--no_dedup",       action="store_true")
    args = p.parse_args()

    cfg = Config(
        video_dir           = args.video_dir,
        output_dir          = args.output_dir,
        fps                 = args.fps,
        blur_threshold      = args.blur_threshold,
        min_brightness      = args.min_brightness,
        max_brightness      = args.max_brightness,
        resize_width        = args.resize,
        use_hw_accel        = not args.no_hw,
        hw_backend          = args.hw_backend,
        hash_diff_threshold = 0 if args.no_dedup else 6,
    )

    run(cfg)
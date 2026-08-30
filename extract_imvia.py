"""
Extract pose features from IMVIA / Le2i Fall Dataset videos.

Expects the extracted dataset under dataset/imvia/:
    dataset/imvia/<Room>/<Room>/Videos/video (N).avi
    dataset/imvia/<Room>/<Room>/Annotation_files/video (N).txt

Annotation format (per-frame lines: frame,state,x1,y1,x2,y2):
    state 0 -> no person         (skipped)
    state 1 -> normal activity   -> "Standing"
    state 8 -> falling           -> "Falling"
    state 7 -> lying after fall  -> "Lying Down"
The first two lines of the annotation hold fall begin / fall end frames
(some files omit them -> all frames treated as not-falling).

Output:
    features/imvia_features.npz   (X: Nx25 pose features, y: N class codes)
    features/imvia_info.json      (class names, feature names, counts)
Optional --save-frames dumps labelled JPEGs to dataset/imvia_frames/<Class>/.
"""
import argparse
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np

from extract_feature import extract_landmarks, extract_features, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExtractImvia")

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "dataset"
IMVIA_DIR = DATA_DIR / "imvia"
FEATURES_DIR = PROJECT_DIR / "features"

CLASS_NAMES = ["Standing", "Falling", "Lying Down"]
CLASS_CODE = {c: i for i, c in enumerate(CLASS_NAMES)}
STATE_TO_CLASS = {1: "Standing", 8: "Falling", 7: "Lying Down"}

VIDEO_EXTS = ("*.avi", "*.mp4", "*.mov", "*.mkv")


def _number_from_name(name: str):
    m = re.search(r"video\s*\(?(\d+)\)?", name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def find_videos():
    videos = {}
    for ext in VIDEO_EXTS:
        for v in IMVIA_DIR.glob(f"**/{ext}"):
            n = _number_from_name(v.stem)
            if n is not None:
                videos[n] = v
    return videos


def find_annotations():
    anns = {}
    for a in IMVIA_DIR.glob("**/*.txt"):
        if a.parent.name.lower().startswith("annotation"):
            n = _number_from_name(a.stem)
            if n is not None:
                anns[n] = a
    return anns


def parse_annotation(path: Path):
    """Return (fall_begin, fall_end, {frame_no: state})."""
    with open(path) as f:
        raw = f.read().splitlines()
    fall_begin = fall_end = None
    state = {}
    i = 0
    while i < len(raw) and len(raw[i].strip()) == 0:
        i += 1
    if i < len(raw):
        first = re.split(r"[,\s]+", raw[i].strip())
        nums = [int(x) for x in first if x.strip()]
        if len(nums) >= 2:
            fall_begin, fall_end = nums[0], nums[1]
            i += 1
        elif len(nums) == 1 and i + 1 < len(raw):
            nxt = re.split(r"[,\s]+", raw[i + 1].strip())
            nxt_nums = [int(x) for x in nxt if x.strip()]
            if len(nxt_nums) >= 1:
                fall_begin, fall_end = nums[0], nxt_nums[0]
                i += 2
    for line in raw[i:]:
        parts = re.split(r"[,\s]+", line.strip())
        if len(parts) < 2:
            continue
        try:
            frame, st = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        state[frame] = st
    return fall_begin, fall_end, state


def process_video(video_path: Path, fall_begin, fall_end, state, frame_step, save_frames, counts, limit_per_class):
    X, y, used = [], [], 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"  Cannot open video {video_path.name}, skipping")
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=int)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_step == 0:
            fno = idx + 1
            st = state.get(fno)
            if st is None:
                st = 1 if (fall_begin is None or fno < fall_begin or fno > fall_end) else 8
            cls_name = STATE_TO_CLASS.get(st)
            if cls_name is None:
                idx += 1
                continue
            code = CLASS_CODE[cls_name]
            landmarks = extract_landmarks(frame)
            if landmarks is None:
                idx += 1
                continue
            feats = extract_features(landmarks)
            if np.any(np.isnan(feats)):
                idx += 1
                continue
            X.append(feats)
            y.append(code)
            used += 1
            counts[cls_name] = counts.get(cls_name, 0) + 1
            if save_frames:
                cls_dir = DATA_DIR / "imvia_frames" / cls_name
                cls_dir.mkdir(parents=True, exist_ok=True)
                out = cls_dir / f"{video_path.stem.replace(' ', '_')}_f{fno:05d}.jpg"
                cv2.imwrite(str(out), frame)
        idx += 1
    cap.release()
    logger.info(f"  {video_path.name}: {used} usable frames")
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Extract pose features from IMVIA videos")
    parser.add_argument("--frame-step", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N videos")
    parser.add_argument("--save-frames", action="store_true", help="Also dump labelled JPEGs")
    parser.add_argument("--only", type=int, nargs="*", default=[], help="Only these video numbers")
    args = parser.parse_args()

    if not IMVIA_DIR.exists():
        logger.error(f"{IMVIA_DIR} not found. Extract the zip into dataset/imvia first.")
        return

    videos = find_videos()
    anns = find_annotations()
    logger.info(f"Found {len(videos)} videos, {len(anns)} annotation files")

    ordered = sorted(videos.items())
    if args.limit:
        ordered = ordered[: args.limit]
    if args.only:
        ordered = [v for v in ordered if v[0] in args.only]

    X_list, y_list = [], []
    counts = {}
    per_video = {}
    for n, video_path in ordered:
        ann_path = anns.get(n)
        if ann_path is None:
            logger.warning(f"  video {n}: no annotation file, skipping")
            continue
        fall_begin, fall_end, state = parse_annotation(ann_path)
        Xv, yv = process_video(video_path, fall_begin, fall_end, state,
                               args.frame_step, args.save_frames, counts, None)
        if len(Xv) == 0:
            continue
        X_list.append(Xv)
        y_list.append(yv)
        per_video[video_path.name] = int(len(yv))

    if not X_list:
        logger.error("No usable pose features extracted.")
        return

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    logger.info(f"Total pose feature samples: {len(X)}")
    for i, c in enumerate(CLASS_NAMES):
        logger.info(f"  {c}: {int(np.sum(y == i))}")

    FEATURES_DIR.mkdir(exist_ok=True)
    np.savez(FEATURES_DIR / "imvia_features.npz", X=X, y=y)
    info = {
        "class_names": CLASS_NAMES,
        "feature_names": FEATURE_NAMES,
        "counts": {c: int(np.sum(y == CLASS_CODE[c])) for c in CLASS_NAMES},
        "per_video": per_video,
        "frame_step": args.frame_step,
        "source": "IMVIA / Le2i (Kaggle tuyenldvn/falldataset-imvia)",
    }
    with open(FEATURES_DIR / "imvia_info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Saved -> {FEATURES_DIR / 'imvia_features.npz'}")
    logger.info(f"Saved -> {FEATURES_DIR / 'imvia_info.json'}")


if __name__ == "__main__":
    main()

"""Build a sequence dataset for the LSTM/GRU fall detector.

Each sample is SEQUENCE_LEN consecutive frame-feature vectors drawn from ONE
video source, so the network learns the motion pattern of a fall (fast drop
to horizontal, then stillness) instead of classifying single poses.

Frame-level fall labels use exactly the same rules as extract_feature.py:
  - UR Fall       : frames inside [onset-12, onset+60] (from height.csv)
  - IMVIA         : per-frame annotations (label==1 == fall)
  - HMDB51 Falling: frames >= 40% of the clip (falls happen after setup)
  - other HMDB    : no fall
  - Recorded      : no fall (Standing/Sitting footage)

A window is labeled fall when the majority of its frames are fall frames.
Groups are video/clip ids so the train/val/test split never leaks a clip.

Output: features/sequences.npz
"""
import sys
import json
import logging
from pathlib import Path

import cv2
import numpy as np

import config
from features import extract_features, FEATURE_NAMES, TemporalWindowExtractor
from extract_feature import (
    extract_landmarks, load_urfall_height, find_fall_onset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildSequences")

SEQ_LEN = config.SEQUENCE_LEN
STRIDE = config.SEQUENCE_STRIDE
WINDOW = config.FEATURE_WINDOW
FEATURE_DIM = len(FEATURE_NAMES)
SAMPLE_EVERY = 2          # pose-detect every 2nd frame: 2x speedup, window still ~2.5s of motion
MAX_IMVIA_VIDEOS = 40
MAX_HMDB_CLIPS = 24


def _window_label(labels):
    """Majority-vote fall label for a window of frame labels."""
    return int(np.mean(labels) >= 0.5)


def _sliding_windows(feats, labels, clip_id):
    """Yield (sequence, label, group) for every stride-spaced window."""
    n = len(feats)
    out = []
    for start in range(0, n - SEQ_LEN + 1, STRIDE):
        seq = np.stack(feats[start:start + SEQ_LEN])   # (SEQ_LEN, D)
        lab = _window_label(labels[start:start + SEQ_LEN])
        out.append((seq, lab, clip_id))
    return out


def _read_imvia_annotation(txt_path):
    """frame_index -> 1 (fall) / 0. Rows look like 'frame,label,x1,y1,x2,y2'."""
    ann = {}
    if not txt_path.exists():
        return ann
    for line in txt_path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                ann[int(parts[0])] = 1 if parts[1] == "1" else 0
            except ValueError:
                continue
    return ann


def collect():
    logger.info("=" * 60)
    logger.info("Building sequence dataset (this runs pose detection over videos)")
    logger.info("=" * 60)
    all_seqs, all_labs, all_groups = [], [], []
    src_counts = {}

    # ---- UR Fall ---------------------------------------------------
    urfall = config.DATASET_DIR / "urfall" / "fall"
    if urfall.exists():
        n_w = 0
        for seq in sorted(urfall.iterdir()):
            if not seq.is_dir():
                continue
            rows = load_urfall_height(seq / "height.csv")
            onset = find_fall_onset(rows)
            if onset is None:
                continue
            files = sorted((seq / "rgb").rglob("*.png"))
            if not files:
                continue
            ext = TemporalWindowExtractor(window=WINDOW)
            feats, labels = [], []
            for f in files:
                try:
                    fr = int(f.stem.rsplit("-", 1)[-1])
                except ValueError:
                    continue
                if fr % SAMPLE_EVERY != 0:
                    continue
                img = cv2.imread(str(f))
                if img is None:
                    continue
                lm = extract_landmarks(img)
                if lm is not None:
                    feat = extract_features(lm, ext.temporal_features())
                    if not np.any(np.isnan(feat)):
                        feats.append(feat)
                        labels.append(1 if (onset - 12 <= fr <= onset + 60) else 0)
                ext.push(lm, fr / 30.0) if lm is not None else None
            ext.reset()
            for seq_arr, lab, g in _sliding_windows(feats, labels, f"urfall:{seq.name}"):
                all_seqs.append(seq_arr); all_labs.append(lab); all_groups.append(g)
                n_w += 1
        logger.info(f"  UR Fall windows: {n_w}")
        src_counts["urfall"] = n_w

    # ---- IMVIA -----------------------------------------------------
    imvia = config.DATASET_DIR / "imvia"
    if imvia.exists():
        n_w = 0
        for vdir in sorted(imvia.rglob("Videos")):
            if not vdir.is_dir():
                continue
            candidates = list(vdir.parent.glob("*nnotation*"))
            ann_dir = candidates[0] if candidates else None
            video_files = [v for v in sorted(vdir.glob("*.avi")) + sorted(vdir.glob("*.mp4"))][:MAX_IMVIA_VIDEOS]
            for video in video_files:
                ann = {}
                if ann_dir is not None:
                    txt = ann_dir / f"{video.stem}.txt"
                    if txt.exists():
                        ann = _read_imvia_annotation(txt)
                cap = cv2.VideoCapture(str(video))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                feats, labels = [], []
                ext = TemporalWindowExtractor(window=WINDOW)
                idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % SAMPLE_EVERY != 0:
                        idx += 1
                        continue
                    lm = extract_landmarks(frame)
                    if lm is not None:
                        feat = extract_features(lm, ext.temporal_features())
                        if not np.any(np.isnan(feat)):
                            feats.append(feat)
                            labels.append(ann.get(idx, 0))
                    ext.push(lm, idx / max(1.0, fps)) if lm is not None else None
                    idx += 1
                cap.release(); ext.reset()
                for seq_arr, lab, g in _sliding_windows(feats, labels, f"imvia:{video.stem}"):
                    all_seqs.append(seq_arr); all_labs.append(lab); all_groups.append(g)
                    n_w += 1
        logger.info(f"  IMVIA windows: {n_w}")
        src_counts["imvia"] = n_w

    # ---- HMDB51 ----------------------------------------------------
    hmdb = config.DATASET_DIR / "hmdb51"
    if hmdb.exists():
        n_w = 0
        for d in sorted(hmdb.iterdir()):
            if not d.is_dir():
                continue
            is_falling = d.name == "Falling"
            for clip in sorted(list(d.glob("*.avi")) + list(d.glob("*.mp4")))[:MAX_HMDB_CLIPS]:
                cap = cv2.VideoCapture(str(clip))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                clip_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                onset = int(clip_len * 0.4) if is_falling else 0
                feats, labels = [], []
                ext = TemporalWindowExtractor(window=WINDOW)
                idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % SAMPLE_EVERY != 0:
                        idx += 1
                        continue
                    lm = extract_landmarks(frame)
                    if lm is not None:
                        feat = extract_features(lm, ext.temporal_features())
                        if not np.any(np.isnan(feat)):
                            feats.append(feat)
                            labels.append(1 if (is_falling and idx >= onset) else 0)
                    ext.push(lm, idx / max(1.0, fps)) if lm is not None else None
                    idx += 1
                cap.release(); ext.reset()
                for seq_arr, lab, g in _sliding_windows(feats, labels, f"hmdb:{d.name}:{clip.stem}"):
                    all_seqs.append(seq_arr); all_labs.append(lab); all_groups.append(g)
                    n_w += 1
        logger.info(f"  HMDB51 windows: {n_w}")
        src_counts["hmdb51"] = n_w

    # ---- Recorded videos --------------------------------------------
    rec = config.RECORDED_DIR
    if rec.exists():
        n_w = 0
        for d in sorted(rec.iterdir()):
            if not d.is_dir():
                continue
            for clip in sorted(list(d.glob("*.avi")) + list(d.glob("*.mp4"))):
                cap = cv2.VideoCapture(str(clip))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                feats, labels = [], []
                ext = TemporalWindowExtractor(window=WINDOW)
                idx = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % SAMPLE_EVERY != 0:
                        idx += 1
                        continue
                    lm = extract_landmarks(frame)
                    if lm is not None:
                        feat = extract_features(lm, ext.temporal_features())
                        if not np.any(np.isnan(feat)):
                            feats.append(feat)
                            labels.append(1 if d.name == "Falling" else 0)
                    ext.push(lm, idx / max(1.0, fps)) if lm is not None else None
                    idx += 1
                cap.release(); ext.reset()
                for seq_arr, lab, g in _sliding_windows(feats, labels, f"rec:{d.name}:{clip.stem}"):
                    all_seqs.append(seq_arr); all_labs.append(lab); all_groups.append(g)
                    n_w += 1
        logger.info(f"  Recorded windows: {n_w}")
        src_counts["recorded"] = n_w

    X = np.stack(all_seqs).astype(np.float32)
    y = np.array(all_labs, dtype=int)
    groups = np.array(all_groups, dtype=object)
    logger.info(f"Total sequences: {len(X)} | fall={int(y.sum())} no-fall={int(len(y)-y.sum())}")
    return X, y, groups, src_counts


def main():
    from sklearn.model_selection import GroupShuffleSplit

    X, y, groups, src_counts = collect()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr0, X_te, y_tr0, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    g_tr0, g_te = groups[tr_idx], groups[te_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42)
    tr2, va_idx = next(gss2.split(X_tr0, y_tr0, g_tr0))
    X_tr, X_va, y_tr, y_va = X_tr0[tr2], X_tr0[va_idx], y_tr0[tr2], y_tr0[va_idx]
    logger.info(f"Split -> train {len(X_tr)} | val {len(X_va)} | test {len(X_te)} (group split)")

    np.savez(config.FEATURES_DIR / "sequences.npz",
             X_train=X_tr, y_train=y_tr, g_train=g_tr0[tr2],
             X_val=X_va, y_val=y_va, g_val=g_tr0[va_idx],
             X_test=X_te, y_test=y_te, g_test=g_te)
    (config.FEATURES_DIR / "sequences_info.json").write_text(json.dumps({
        "seq_len": SEQ_LEN, "stride": STRIDE, "n_features": FEATURE_DIM,
        "feature_names": FEATURE_NAMES, "source_counts": src_counts,
    }, indent=2))
    logger.info(f"Saved -> {config.FEATURES_DIR / 'sequences.npz'}")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from features import (
    extract_features, FEATURE_NAMES, TEMPORAL_FEATURE_NAMES, TemporalWindowExtractor,
)

ACTIVITIES = [
    "Standing", "Sitting", "Walking", "Running", "Falling",
    "Lying Down", "Bending", "Squatting", "Jumping", "Climbing Stairs",
    "Crawling", "Kneeling", "Crouching", "Getting Up",
]
ACTIVITY_MAP = {a: i for i, a in enumerate(ACTIVITIES)}

# Whitelist of activities the final model keeps (curated to the ones that work:
# the activities users actually recorded, plus the essential Fall class).
KEEP_ACTIVITIES = [
    "Falling", "Crouching", "Kneeling", "Lying Down",
    "Sitting", "Standing", "Walking",
]

# Dataset folder names that should be merged into an existing class.
_ACTIVITY_REMAP = {
    "Sitting Down": "Sitting",
    "Sitting on Floor": "Sitting",
}

# Group ids keep frames from the same clip/sequence together so the
# train/val/test split has no temporal leakage.
def _synthetic_budget(act, n_real):
    """Synthetic supplement only where real data is thin (real poses are better)."""
    if act not in _POSE_GENERATORS:
        return 0
    if n_real < 150:
        return 300
    if n_real < 350:
        return 150
    return 0


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

_MP_MODEL_PATH = "models/pose_landmarker_lite.task"
_base_options = python.BaseOptions(model_asset_path=_MP_MODEL_PATH)
_options = PoseLandmarkerOptions(
    base_options=_base_options, running_mode=RunningMode.IMAGE,
    min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.4,
    num_poses=1, output_segmentation_masks=False,
)
_pose_landmarker = PoseLandmarker.create_from_options(_options)

# Pose inference cost scales with pixels; downscale oversized frames (photos
# from the datasets are often 1080p+) before detection. Landmarks are
# normalized to [0,1], so this does not change the feature values.
_POSE_MAX_DIM = 640


def extract_landmarks(image):
    h, w = image.shape[:2]
    if max(h, w) > _POSE_MAX_DIM:
        scale = _POSE_MAX_DIM / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb, dtype=np.uint8))
    result = _pose_landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return None
    arr = []
    for lm in result.pose_landmarks[0]:
        arr.extend([lm.x, lm.y, lm.z, lm.visibility if lm.visibility else 0.0])
    return np.array(arr, dtype=np.float32)


def _frame_groups(files, strategy="individual", block_size=30):
    """Return group ids that prevent adjacent video frames leaking between splits.

    A random image should remain a group of one.  Frame-export datasets use
    numeric filenames (for example ``1.jpg`` .. ``1200.jpg``), so nearby
    images are placed in blocks.  Webcam captures include epoch timestamps;
    a gap of more than two seconds marks a new recording session.
    """
    if strategy == "individual":
        return [str(f) for f in files]

    groups = []
    if strategy == "numeric_blocks":
        for f in files:
            digits = re.findall(r"\d+", Path(f).stem)
            if digits:
                block = (int(digits[-1]) - 1) // block_size
                groups.append(f"{Path(f).parent}:block:{block}")
            else:
                groups.append(str(f))
        return groups

    if strategy == "recorded_sessions":
        session, last_timestamp = 0, None
        for f in files:
            match = re.match(r"frame_(\d+)_\d+$", Path(f).stem)
            timestamp = int(match.group(1)) if match else None
            if timestamp is None or (last_timestamp is not None and timestamp - last_timestamp > 2000):
                session += 1
            groups.append(f"{Path(f).parent}:session:{session}")
            last_timestamp = timestamp
        return groups

    raise ValueError(f"Unknown frame grouping strategy: {strategy}")


def process_dataset_images(image_dir, label, limit=None, group_strategy="individual"):
    from glob import glob
    X, y, groups = [], [], []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob(str(Path(image_dir) / ext)))
        files.extend(glob(str(Path(image_dir) / "**" / ext), recursive=True))
    files = sorted(set(files))
    if limit:
        files = files[:limit]
    group_ids = _frame_groups(files, strategy=group_strategy)

    detected = 0
    for f, group_id in zip(files, group_ids):
        img = cv2.imread(f)
        if img is None:
            continue
        landmarks = extract_landmarks(img)
        if landmarks is None:
            continue
        feats = extract_features(landmarks)          # static image: no motion
        if not np.any(np.isnan(feats)):
            X.append(feats)
            y.append(label)
            groups.append(group_id)
            detected += 1
    return np.array(X, dtype=np.float32), np.array(y, dtype=int), groups, detected, len(files)


# ──────────────────────────────────────────────────────────────
# UR FALL (dataset/urfall) + HMDB51 (dataset/hmdb51) ingestion.
# UR Fall provides real RGB frames of people falling; the body-height
# csv locates the fall moment so pre-fall frames are not mislabeled.
# HMDB51 provides real clips for Walking/Running/Jumping/Climbing
# Stairs/Sitting Down/Getting Up/Falling.
# ──────────────────────────────────────────────────────────────
def load_urfall_height(csv_path):
    """Parse 'frame,time,height' rows from a UR Fall data csv."""
    rows = []
    if not Path(csv_path).exists():
        return rows
    for line in Path(csv_path).read_text().strip().splitlines():
        parts = line.replace(";", ",").split(",")
        if len(parts) >= 3:
            try:
                rows.append((int(float(parts[0])), float(parts[2])))
            except ValueError:
                continue
    return rows


def find_fall_onset(rows, window=5):
    """First frame where smoothed body height drops far below its baseline."""
    if not rows:
        return None
    baseline = float(np.median([h for _, h in rows[:20]]))
    hs = np.array([h for _, h in rows], dtype=np.float64)
    k = min(window, len(hs))
    if k > 1:
        hs = np.convolve(hs, np.ones(k) / k, mode="same")
    for i, (fr, _) in enumerate(rows):
        if hs[i] < baseline * 0.70:
            return fr
    for i, (fr, _) in enumerate(rows):
        if (i + 3 < len(rows) and hs[i] < baseline * 0.78
                and all(hs[i:i + 4] < baseline * 0.92)):
            return fr
    return None


def process_urfall_falls(fall_root, label, sample_every=3, pre=12, post=60):
    """Label the fall-event window of each UR Fall sequence as Falling.
    Temporal features are computed per sequence (window extractor reset between
    sequences) so the model learns the downward motion, not just the pose."""
    X = []
    groups = []
    total = 0
    for seq in sorted(Path(fall_root).iterdir()):
        if not seq.is_dir():
            continue
        rgb = seq / "rgb"
        rows = load_urfall_height(seq / "height.csv")
        onset = find_fall_onset(rows)
        if onset is None:
            continue
        files = sorted(rgb.rglob("*.png"))
        ext = TemporalWindowExtractor()
        seq_start = files[0] if files else None
        for f in files:
            try:
                fr = int(f.stem.rsplit("-", 1)[-1])
            except ValueError:
                continue
            if not (onset - pre <= fr <= onset + post):
                continue
            img = cv2.imread(str(f))
            if img is None:
                continue
            landmarks = extract_landmarks(img)
            if landmarks is None:
                continue
            ext.push(landmarks, fr / 30.0)            # UR Fall captured at 30 fps
            if fr % sample_every != 0:
                continue
            feats = extract_features(landmarks, ext.temporal_features())
            if not np.any(np.isnan(feats)):
                X.append(feats)
                groups.append(seq.name)               # whole sequence stays together
                total += 1
        ext.reset()
    return np.array(X, dtype=np.float32), np.full(len(X), label, dtype=int), groups, total


def process_hmdb_clips(class_dir, label, sample_every=6, max_clips=15, start_frac=0.0):
    """Sample real frames from clips; each clip keeps its own temporal window
    and a shared group id so the train/test split never sees the same video.

    start_frac: for "Falling" clips, only frames at/after this fraction of the
    clip are labeled as the class. HMDB fall clips begin with the subject doing
    ADL (walking/sitting) and only later fall, so earlier frames would be
    mislabeled as falls.
    """
    X = []
    groups = []
    total = 0
    clips = sorted(Path(class_dir).glob("*.avi")) + sorted(Path(class_dir).glob("*.mp4"))
    for clip in clips[:max_clips]:
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        clip_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        skip_until = int(clip_len * start_frac) if (start_frac and clip_len > 0) else 0
        ext = TemporalWindowExtractor()
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            landmarks = extract_landmarks(frame)
            if landmarks is not None:
                ext.push(landmarks, idx / fps)
                if idx >= skip_until and idx % sample_every == 0:
                    feats = extract_features(landmarks, ext.temporal_features())
                    if not np.any(np.isnan(feats)):
                        X.append(feats)
                        groups.append(clip.stem)      # clip-level group
                        total += 1
            idx += 1
        cap.release()
        ext.reset()
    return np.array(X, dtype=np.float32), np.full(len(X), label, dtype=int), groups, total


def process_imvia_falls(imvia_root, label, sample_every=8, max_videos=15):
    """IMVIA is a CCTV fall dataset: every video is a real fall event.
    Frames are pushed through a temporal window per video so the model learns
    the descent motion, and each video keeps its own group id."""
    X = []
    groups = []
    total = 0
    video_dirs = [p for p in sorted(Path(imvia_root).rglob("Videos"))
                  if p.is_dir()] if Path(imvia_root).exists() else []
    count = 0
    for vdir in video_dirs:
        for video in sorted(vdir.glob("*.avi")) + sorted(vdir.glob("*.mp4")):
            if count >= max_videos:
                break
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            ext = TemporalWindowExtractor()
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                landmarks = extract_landmarks(frame)
                if landmarks is not None:
                    ext.push(landmarks, idx / fps)
                    if idx % sample_every == 0:
                        feats = extract_features(landmarks, ext.temporal_features())
                        if not np.any(np.isnan(feats)):
                            X.append(feats)
                            groups.append(video.stem)
                            total += 1
                idx += 1
            cap.release()
            ext.reset()
            count += 1
    return np.array(X, dtype=np.float32), np.full(len(X), label, dtype=int), groups, total


# ──────────────────────────────────────────────────────────────
# SYNTHETIC POSE GENERATORS (one per activity)
# Coordinates follow MediaPipe convention: x,y in [0,1], y DOWN.
# Used to supplement classes that have little or no real data.
# ──────────────────────────────────────────────────────────────
def _build_pose(base, label):
    """Add small per-landmark noise so each class generalizes."""
    out = base.copy()
    out += np.random.normal(0, 0.002, out.shape).astype(np.float32)
    out = np.clip(out, 0, 1)
    return out


def _set(lm, idx, x, y, z=0.0, vis=0.95):
    lm[idx * 4] = x; lm[idx * 4 + 1] = y
    lm[idx * 4 + 2] = z; lm[idx * 4 + 3] = vis


def _standing_pose(rng):
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.4, 0.6); head_y = rng.uniform(0.08, 0.18)
    torso = rng.uniform(0.22, 0.28); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18); leg = rng.uniform(0.38, 0.46)
    sh_y = head_y + 0.09; hip_y = sh_y + torso; knee_y = hip_y + leg * 0.55; ank_y = hip_y + leg
    _set(lm, 0, tx, head_y, vis=0.99)
    _set(lm, 11, tx - sw/2, sh_y, vis=0.99); _set(lm, 12, tx + sw/2, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    _set(lm, 25, tx - hw/2, knee_y, vis=0.96); _set(lm, 26, tx + hw/2, knee_y, vis=0.96)
    _set(lm, 27, tx - hw/2, ank_y, vis=0.95); _set(lm, 28, tx + hw/2, ank_y, vis=0.95)
    _set(lm, 29, tx - hw/2 - 0.01, ank_y + 0.02, vis=0.93)
    _set(lm, 30, tx + hw/2 + 0.01, ank_y + 0.02, vis=0.93)
    _set(lm, 31, tx - hw/2 - 0.02, ank_y + 0.04, vis=0.92)
    _set(lm, 32, tx + hw/2 + 0.02, ank_y + 0.04, vis=0.92)
    arm = rng.uniform(-0.03, 0.03)
    _set(lm, 13, tx - sw/2 + arm, sh_y + torso * 0.45, vis=0.95)
    _set(lm, 14, tx + sw/2 - arm, sh_y + torso * 0.45, vis=0.95)
    _set(lm, 15, tx - sw/2 + arm*0.5, sh_y + torso * 0.9, vis=0.93)
    _set(lm, 16, tx + sw/2 - arm*0.5, sh_y + torso * 0.9, vis=0.93)
    return lm


def _sitting_pose(rng):
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.4, 0.6); head_y = rng.uniform(0.08, 0.20)
    torso = rng.uniform(0.18, 0.24); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18); shin = rng.uniform(0.18, 0.26)
    sh_y = head_y + 0.08; hip_y = sh_y + torso; kfwd = rng.uniform(0.05, 0.12)
    _set(lm, 0, tx, head_y, vis=0.99)
    _set(lm, 11, tx - sw/2, sh_y, vis=0.99); _set(lm, 12, tx + sw/2, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.90); _set(lm, 24, tx + hw/2, hip_y, vis=0.90)
    _set(lm, 25, tx - hw/2 + kfwd, hip_y + 0.03, vis=0.85)
    _set(lm, 26, tx + hw/2 + kfwd, hip_y + 0.03, vis=0.85)
    _set(lm, 27, tx - hw/2 + kfwd, hip_y + 0.03 + shin, vis=0.80)
    _set(lm, 28, tx + hw/2 + kfwd, hip_y + 0.03 + shin, vis=0.80)
    _set(lm, 29, tx - hw/2 + kfwd - 0.01, hip_y + 0.05 + shin, vis=0.75)
    _set(lm, 30, tx + hw/2 + kfwd + 0.01, hip_y + 0.05 + shin, vis=0.75)
    _set(lm, 31, tx - hw/2 + kfwd - 0.02, hip_y + 0.07 + shin, vis=0.74)
    _set(lm, 32, tx + hw/2 + kfwd + 0.02, hip_y + 0.07 + shin, vis=0.74)
    _set(lm, 13, tx - sw/2 - 0.04, sh_y + torso*0.4, vis=0.94)
    _set(lm, 14, tx + sw/2 + 0.04, sh_y + torso*0.4, vis=0.94)
    _set(lm, 15, tx - sw/2 - 0.02, sh_y + torso*0.75, vis=0.92)
    _set(lm, 16, tx + sw/2 + 0.02, sh_y + torso*0.75, vis=0.92)
    return lm


def _walking_pose(rng):
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.4, 0.6); head_y = rng.uniform(0.08, 0.18)
    torso = rng.uniform(0.22, 0.28); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18); leg = rng.uniform(0.38, 0.46)
    step = rng.uniform(0.05, 0.14)
    sh_y = head_y + 0.09; hip_y = sh_y + torso; knee_y = hip_y + leg * 0.52; ank_y = hip_y + leg
    _set(lm, 0, tx, head_y, vis=0.99)
    _set(lm, 11, tx - sw/2, sh_y, vis=0.99); _set(lm, 12, tx + sw/2, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    _set(lm, 25, tx - hw/2 - step, knee_y + rng.uniform(-0.02, 0.02), vis=0.96)
    _set(lm, 26, tx + hw/2 + step*0.4, knee_y + rng.uniform(-0.02, 0.02), vis=0.96)
    _set(lm, 27, tx - hw/2 - step*1.1, ank_y, vis=0.95)
    _set(lm, 28, tx + hw/2 + step*0.2, ank_y + rng.uniform(0.0, 0.04), vis=0.95)
    _set(lm, 29, tx - hw/2 - step*1.1 - 0.01, ank_y + 0.02, vis=0.93)
    _set(lm, 30, tx + hw/2 + step*0.2 + 0.01, ank_y + 0.05, vis=0.93)
    _set(lm, 31, tx - hw/2 - step*1.1 - 0.02, ank_y + 0.04, vis=0.92)
    _set(lm, 32, tx + hw/2 + step*0.2 + 0.02, ank_y + 0.07, vis=0.92)
    arm = rng.uniform(0.02, 0.09)
    elb_y = sh_y + torso * 0.42; wrs_y = sh_y + torso * 0.82
    _set(lm, 13, tx - sw/2 + arm, elb_y, vis=0.95); _set(lm, 14, tx + sw/2 - arm, elb_y, vis=0.95)
    _set(lm, 15, tx - sw/2 - arm*0.4, wrs_y, vis=0.93); _set(lm, 16, tx + sw/2 + arm*0.4, wrs_y, vis=0.93)
    return lm


def _running_pose(rng):
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.4, 0.6); head_y = rng.uniform(0.08, 0.18)
    torso = rng.uniform(0.22, 0.28); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18); leg = rng.uniform(0.38, 0.46)
    fwd = rng.uniform(0.10, 0.20)         # forward lean
    sh_y = head_y + 0.09 + fwd*0.2; hip_y = sh_y + torso - fwd*0.1
    knee_y = hip_y + leg * 0.52; ank_y = hip_y + leg
    _set(lm, 0, tx, head_y, vis=0.99)
    _set(lm, 11, tx - sw/2 - fwd*0.3, sh_y, vis=0.99)
    _set(lm, 12, tx + sw/2 - fwd*0.3, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    # front leg tucked, rear leg extended
    _set(lm, 25, tx - hw/2 + fwd*0.5, knee_y - 0.10, vis=0.96)
    _set(lm, 26, tx + hw/2 - fwd*0.6, knee_y + 0.02, vis=0.96)
    _set(lm, 27, tx - hw/2 + fwd*0.4, ank_y - 0.16, vis=0.95)
    _set(lm, 28, tx + hw/2 - fwd*0.7, ank_y - 0.05, vis=0.95)
    _set(lm, 29, tx - hw/2 + fwd*0.4 - 0.01, ank_y - 0.14, vis=0.93)
    _set(lm, 30, tx + hw/2 - fwd*0.7 + 0.01, ank_y - 0.03, vis=0.93)
    _set(lm, 31, tx - hw/2 + fwd*0.4 - 0.02, ank_y - 0.12, vis=0.92)
    _set(lm, 32, tx + hw/2 - fwd*0.7 + 0.02, ank_y - 0.01, vis=0.92)
    # arms bent ~90deg pumping
    arm = rng.uniform(0.10, 0.18)
    _set(lm, 13, tx - sw/2 + arm*0.7, sh_y + torso*0.35, vis=0.95)
    _set(lm, 14, tx + sw/2 - arm*0.7, sh_y + torso*0.35, vis=0.95)
    _set(lm, 15, tx - sw/2 + arm, sh_y + torso*0.62, vis=0.93)
    _set(lm, 16, tx + sw/2 - arm, sh_y + torso*0.62, vis=0.93)
    return lm


def _falling_pose(rng):
    """Mid-fall: body rotated towards horizontal, still partially up."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.35, 0.65); ty = rng.uniform(0.25, 0.50)
    angle = rng.uniform(0.35, 1.25)
    body_len = rng.uniform(0.35, 0.50)
    sw = rng.uniform(0.08, 0.16)
    hx = tx - np.sin(angle)*body_len*0.45; hy = ty - np.cos(angle)*body_len*0.05
    _set(lm, 0, hx, hy, vis=0.98)
    sx = tx - np.sin(angle)*body_len*0.15; sy = ty - np.cos(angle)*body_len*0.02
    _set(lm, 11, sx - sw/2, sy, vis=0.97); _set(lm, 12, sx + sw/2, sy, vis=0.97)
    hipx = tx + np.sin(angle)*body_len*0.2; hipy = ty + abs(np.cos(angle))*body_len*0.05
    _set(lm, 23, hipx - sw*0.3, hipy, vis=0.96); _set(lm, 24, hipx + sw*0.3, hipy, vis=0.96)
    k1x = hipx + np.sin(angle)*0.08; k1y = hipy + abs(np.cos(angle))*0.08
    k2x = k1x + rng.uniform(0.04, 0.12); k2y = k1y + rng.uniform(-0.04, 0.06)
    _set(lm, 25, k1x, k1y, vis=0.94); _set(lm, 26, k2x, k2y, vis=0.94)
    a1x = k1x + rng.uniform(0.03, 0.08); a1y = k1y + rng.uniform(0.04, 0.10)
    a2x = k2x + rng.uniform(0.03, 0.10); a2y = k2y + rng.uniform(0.03, 0.10)
    _set(lm, 27, a1x, a1y, vis=0.92); _set(lm, 28, a2x, a2y, vis=0.92)
    _set(lm, 29, a1x + 0.01, a1y + 0.02, vis=0.90); _set(lm, 30, a2x + 0.01, a2y + 0.02, vis=0.90)
    _set(lm, 31, a1x + 0.02, a1y + 0.04, vis=0.89); _set(lm, 32, a2x + 0.02, a2y + 0.04, vis=0.89)
    af = rng.uniform(0.04, 0.14)
    _set(lm, 13, sx - sw/2 - af, sy - af*0.3, vis=0.93)
    _set(lm, 14, sx + sw/2 + af, sy + af*0.5, vis=0.93)
    _set(lm, 15, sx - sw/2 - af*1.3, sy + 0.06, vis=0.91)
    _set(lm, 16, sx + sw/2 + af*1.0, sy + 0.12, vis=0.91)
    return lm


def _lying_pose(rng):
    """Fully horizontal body, torso ~90 deg, legs straight."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    ty = rng.uniform(0.35, 0.62)
    bx = rng.uniform(0.20, 0.45)          # head toward the left
    body_len = rng.uniform(0.45, 0.62)
    sw = rng.uniform(0.10, 0.18)
    _set(lm, 0, bx, ty, vis=0.99)
    _set(lm, 11, bx + body_len*0.12, ty, vis=0.99)
    _set(lm, 12, bx + body_len*0.12, ty + sw*0.4, vis=0.99)
    hipx = bx + body_len*0.34
    _set(lm, 23, hipx, ty - sw*0.3, vis=0.97)
    _set(lm, 24, hipx, ty + sw*0.4, vis=0.97)
    knee = bx + body_len*0.60
    _set(lm, 25, knee, ty - sw*0.3, vis=0.96)
    _set(lm, 26, knee, ty + sw*0.4, vis=0.96)
    ank = bx + body_len*0.90
    _set(lm, 27, ank, ty - sw*0.3, vis=0.95)
    _set(lm, 28, ank, ty + sw*0.4, vis=0.95)
    _set(lm, 29, ank + 0.01, ty - sw*0.3, vis=0.93)
    _set(lm, 30, ank + 0.01, ty + sw*0.4, vis=0.93)
    _set(lm, 31, ank + 0.03, ty - sw*0.3, vis=0.92)
    _set(lm, 32, ank + 0.03, ty + sw*0.4, vis=0.92)
    _set(lm, 13, bx + body_len*0.15, ty - sw*0.4, vis=0.95)
    _set(lm, 14, bx + body_len*0.15, ty + sw*0.6, vis=0.95)
    _set(lm, 15, bx + body_len*0.22, ty - sw*0.4, vis=0.93)
    _set(lm, 16, bx + body_len*0.22, ty + sw*0.6, vis=0.93)
    return lm


def _bending_pose(rng):
    """Straight legs, torso folds forward so shoulders reach near hip/knee height."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.42, 0.58)
    sw = rng.uniform(0.14, 0.20); hw = rng.uniform(0.12, 0.18)
    torso = rng.uniform(0.22, 0.28); leg = rng.uniform(0.38, 0.46)
    hip_y = rng.uniform(0.42, 0.52)
    ank_y = hip_y + leg; knee_y = hip_y + leg * 0.55
    # shoulders fold forward/down: 50-80 deg from vertical (torso_inclination ~100-140)
    fold = rng.uniform(0.90, 1.40)
    sh_x = tx + np.sin(fold) * torso
    sh_y = hip_y - np.cos(fold) * torso
    nose_x = sh_x + np.sin(fold) * 0.10
    nose_y = sh_y - np.cos(fold) * 0.10
    _set(lm, 0, nose_x, nose_y, vis=0.99)
    _set(lm, 11, sh_x - sw/2, sh_y, vis=0.99)
    _set(lm, 12, sh_x + sw/2, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    _set(lm, 25, tx - hw/2, knee_y, vis=0.96); _set(lm, 26, tx + hw/2, knee_y, vis=0.96)
    _set(lm, 27, tx - hw/2, ank_y, vis=0.95); _set(lm, 28, tx + hw/2, ank_y, vis=0.95)
    _set(lm, 29, tx - hw/2 - 0.01, ank_y + 0.02, vis=0.93)
    _set(lm, 30, tx + hw/2 + 0.01, ank_y + 0.02, vis=0.93)
    _set(lm, 31, tx - hw/2 - 0.02, ank_y + 0.04, vis=0.92)
    _set(lm, 32, tx + hw/2 + 0.02, ank_y + 0.04, vis=0.92)
    # arms hang toward the floor (roughly parallel to shins)
    arm_len = rng.uniform(0.28, 0.36)
    _set(lm, 13, sh_x - sw/2 - 0.02, sh_y + torso*0.5, vis=0.95)
    _set(lm, 14, sh_x + sw/2 + 0.02, sh_y + torso*0.5, vis=0.95)
    _set(lm, 15, sh_x - sw/2 - 0.02, sh_y + arm_len, vis=0.93)
    _set(lm, 16, sh_x + sw/2 + 0.02, sh_y + arm_len, vis=0.93)
    return lm


def _squatting_pose(rng):
    """Deep knee bend (60-90 deg), hips low, thighs near horizontal, feet flat."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.42, 0.58); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18)
    torso = rng.uniform(0.22, 0.28)
    hip_y = rng.uniform(0.58, 0.68)         # hips dropped well below standing
    knee_y = hip_y + rng.uniform(0.06, 0.12)
    ank_y = hip_y + rng.uniform(0.18, 0.26)  # ankle below knee, foot flat
    head_y = hip_y - torso + rng.uniform(-0.02, 0.02)
    _set(lm, 0, tx, head_y, vis=0.99)
    _set(lm, 11, tx - sw/2, head_y + 0.10, vis=0.99)
    _set(lm, 12, tx + sw/2, head_y + 0.10, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    # knees far forward of hips and ankles nearly under hips -> deep bend (knee ~60-90 deg)
    kx = tx - hw/2 + rng.uniform(0.08, 0.14)
    ax = tx - hw/2 + rng.uniform(-0.02, 0.03)
    _set(lm, 25, kx, knee_y, vis=0.96)
    _set(lm, 26, kx + hw, knee_y, vis=0.96)
    _set(lm, 27, ax, ank_y, vis=0.95)
    _set(lm, 28, ax + hw, ank_y, vis=0.95)
    _set(lm, 29, ax - 0.02, ank_y + 0.02, vis=0.93)
    _set(lm, 30, ax + hw + 0.02, ank_y + 0.02, vis=0.93)
    _set(lm, 31, ax - 0.03, ank_y + 0.04, vis=0.92)
    _set(lm, 32, ax + hw + 0.03, ank_y + 0.04, vis=0.92)
    # arms forward for balance
    _set(lm, 13, tx - hw/2 - 0.12, hip_y - 0.10, vis=0.95)
    _set(lm, 14, tx + hw/2 + 0.12, hip_y - 0.10, vis=0.95)
    _set(lm, 15, tx - hw/2 - 0.14, hip_y - 0.16, vis=0.93)
    _set(lm, 16, tx + hw/2 + 0.14, hip_y - 0.16, vis=0.93)
    return lm


def _jumping_pose(rng):
    """Airborne: knees bent (tucked ~110-140 deg), feet raised, arms up."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.4, 0.6); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18); torso = rng.uniform(0.22, 0.28)
    sh_y = rng.uniform(0.30, 0.42); head_y = sh_y - 0.09
    hip_y = sh_y + torso
    _set(lm, 0, tx, head_y, vis=0.99)
    _set(lm, 11, tx - sw/2, sh_y, vis=0.99); _set(lm, 12, tx + sw/2, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    # knees bent: hips pulled up, feet trail below-behind
    thigh = rng.uniform(0.10, 0.16)          # thigh shorter projection = bent knee
    shin = rng.uniform(0.10, 0.18)
    knee_y = hip_y + thigh
    ank_y = knee_y + shin
    kx = tx - hw/2 + rng.uniform(-0.04, 0.02)
    _set(lm, 25, kx, knee_y, vis=0.96)
    _set(lm, 26, kx + hw, knee_y, vis=0.96)
    _set(lm, 27, kx - 0.04, ank_y, vis=0.95)
    _set(lm, 28, kx + hw + 0.04, ank_y, vis=0.95)
    _set(lm, 29, kx - 0.05, ank_y + 0.02, vis=0.93)
    _set(lm, 30, kx + hw + 0.05, ank_y + 0.02, vis=0.93)
    _set(lm, 31, kx - 0.06, ank_y + 0.04, vis=0.92)
    _set(lm, 32, kx + hw + 0.06, ank_y + 0.04, vis=0.92)
    # arms raised up
    arm = rng.uniform(0.04, 0.08)
    _set(lm, 13, tx - sw/2 + arm, sh_y - 0.12, vis=0.95)
    _set(lm, 14, tx + sw/2 - arm, sh_y - 0.12, vis=0.95)
    _set(lm, 15, tx - sw/2 + arm*0.4, sh_y - 0.18, vis=0.93)
    _set(lm, 16, tx + sw/2 - arm*0.4, sh_y - 0.18, vis=0.93)
    return lm


def _stairs_pose(rng):
    """One foot elevated on a step: asymmetric knees/ankles."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.42, 0.58); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18); torso = rng.uniform(0.22, 0.28)
    leg = rng.uniform(0.38, 0.46)
    hip_y = rng.uniform(0.40, 0.48); sh_y = hip_y - torso
    head_y = sh_y - 0.09
    _set(lm, 0, tx, head_y, vis=0.99)
    _set(lm, 11, tx - sw/2, sh_y, vis=0.99); _set(lm, 12, tx + sw/2, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    # left leg straight on ground, right leg bent up onto step
    _set(lm, 25, tx - hw/2, hip_y + leg*0.55, vis=0.96)      # left knee
    _set(lm, 26, tx + hw/2 + 0.10, hip_y + 0.06, vis=0.96)   # right knee raised
    _set(lm, 27, tx - hw/2, hip_y + leg, vis=0.95)           # left ankle on floor
    _set(lm, 28, tx + hw/2 + 0.14, hip_y + 0.12, vis=0.95)   # right ankle on step
    _set(lm, 29, tx - hw/2 - 0.01, hip_y + leg + 0.02, vis=0.93)
    _set(lm, 30, tx + hw/2 + 0.14, hip_y + 0.14, vis=0.93)
    _set(lm, 31, tx - hw/2 - 0.02, hip_y + leg + 0.04, vis=0.92)
    _set(lm, 32, tx + hw/2 + 0.14, hip_y + 0.16, vis=0.92)
    arm = rng.uniform(0.04, 0.08)
    _set(lm, 13, tx - sw/2 + arm, sh_y + torso*0.4, vis=0.95)
    _set(lm, 14, tx + sw/2 - arm, sh_y + torso*0.4, vis=0.95)
    _set(lm, 15, tx - sw/2 - arm*0.3, sh_y + torso*0.8, vis=0.93)
    _set(lm, 16, tx + sw/2 + arm*0.3, sh_y + torso*0.8, vis=0.93)
    return lm


def _crouching_pose(rng):
    """Deep compact crouch: hips very low, knees deeply bent, torso leaning forward."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.42, 0.58); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18)
    torso = rng.uniform(0.16, 0.22)
    hip_y = rng.uniform(0.62, 0.72)          # hips almost at ankle height
    knee_y = hip_y + rng.uniform(0.04, 0.09)
    ank_y = hip_y + rng.uniform(0.14, 0.20)
    fold = rng.uniform(0.45, 0.85)           # forward lean ~25-50 deg
    sh_x = tx + np.sin(fold) * torso
    sh_y = hip_y - np.cos(fold) * torso
    _set(lm, 0, sh_x, sh_y - 0.08, vis=0.99)
    _set(lm, 11, sh_x - sw/2, sh_y, vis=0.99); _set(lm, 12, sh_x + sw/2, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    kx = tx - hw/2 + rng.uniform(0.05, 0.10)
    ax = tx - hw/2 + rng.uniform(-0.02, 0.03)
    _set(lm, 25, kx, knee_y, vis=0.96); _set(lm, 26, kx + hw, knee_y, vis=0.96)
    _set(lm, 27, ax, ank_y, vis=0.95); _set(lm, 28, ax + hw, ank_y, vis=0.95)
    _set(lm, 29, ax - 0.02, ank_y + 0.02, vis=0.93)
    _set(lm, 30, ax + hw + 0.02, ank_y + 0.02, vis=0.93)
    _set(lm, 31, ax - 0.03, ank_y + 0.04, vis=0.92)
    _set(lm, 32, ax + hw + 0.03, ank_y + 0.04, vis=0.92)
    # hands near knees / floor
    _set(lm, 13, sh_x - sw/2 - 0.03, sh_y + torso*0.45, vis=0.95)
    _set(lm, 14, sh_x + sw/2 + 0.03, sh_y + torso*0.45, vis=0.95)
    _set(lm, 15, tx - hw/2 - 0.06, knee_y + 0.02, vis=0.93)
    _set(lm, 16, tx + hw/2 + 0.06, knee_y + 0.02, vis=0.93)
    return lm


def _kneeling_pose(rng):
    """Knees on the ground, shins flat behind, torso upright."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.42, 0.58); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18); torso = rng.uniform(0.22, 0.28)
    knee_y = rng.uniform(0.58, 0.66)         # knees on floor
    hip_y = knee_y - rng.uniform(0.12, 0.18) # hips above knees
    sh_y = hip_y - torso
    _set(lm, 0, tx, sh_y - 0.09, vis=0.99)
    _set(lm, 11, tx - sw/2, sh_y, vis=0.99); _set(lm, 12, tx + sw/2, sh_y, vis=0.99)
    _set(lm, 23, tx - hw/2, hip_y, vis=0.97); _set(lm, 24, tx + hw/2, hip_y, vis=0.97)
    kx = tx - hw/2 + rng.uniform(0.0, 0.04)
    _set(lm, 25, kx, knee_y, vis=0.96); _set(lm, 26, kx + hw, knee_y, vis=0.96)
    # shins flat on floor behind knees; ankles/feet trail back
    ax = kx - rng.uniform(0.10, 0.16)
    _set(lm, 27, ax, knee_y - 0.02, vis=0.95); _set(lm, 28, ax + hw, knee_y - 0.02, vis=0.95)
    _set(lm, 29, ax - 0.02, knee_y - 0.05, vis=0.93)
    _set(lm, 30, ax + hw + 0.02, knee_y - 0.05, vis=0.93)
    _set(lm, 31, ax - 0.03, knee_y - 0.03, vis=0.92)
    _set(lm, 32, ax + hw + 0.03, knee_y - 0.03, vis=0.92)
    # arms relaxed at sides
    _set(lm, 13, tx - sw/2 - 0.04, sh_y + torso*0.42, vis=0.95)
    _set(lm, 14, tx + sw/2 + 0.04, sh_y + torso*0.42, vis=0.95)
    _set(lm, 15, tx - sw/2 - 0.02, hip_y + 0.02, vis=0.93)
    _set(lm, 16, tx + sw/2 + 0.02, hip_y + 0.02, vis=0.93)
    return lm


def _crawling_pose(rng):
    """Four-point crawl: hands and knees on the floor, body roughly horizontal."""
    lm = np.zeros(33 * 4, dtype=np.float32)
    tx = rng.uniform(0.40, 0.60); sw = rng.uniform(0.14, 0.20)
    hw = rng.uniform(0.12, 0.18)
    floor_y = rng.uniform(0.70, 0.82)
    torso = rng.uniform(0.14, 0.20)          # compressed body height
    fwd = rng.uniform(0.06, 0.14)            # shoulders ahead of hips
    sh_x = tx + fwd; sh_y = floor_y - torso
    hip_x = tx - fwd; hip_y = floor_y - rng.uniform(0.06, 0.12)
    _set(lm, 0, sh_x, sh_y - 0.05, vis=0.99)  # head forward/down
    _set(lm, 11, sh_x - sw/2, sh_y, vis=0.99); _set(lm, 12, sh_x + sw/2, sh_y, vis=0.99)
    _set(lm, 23, hip_x - hw/2, hip_y, vis=0.97); _set(lm, 24, hip_x + hw/2, hip_y, vis=0.97)
    # knees on floor below hips, ankles behind knees on floor
    kx = hip_x - hw/2 + rng.uniform(0.0, 0.04)
    _set(lm, 25, kx, floor_y, vis=0.96); _set(lm, 26, kx + hw, floor_y, vis=0.96)
    ax = kx - rng.uniform(0.10, 0.16)
    _set(lm, 27, ax, floor_y, vis=0.95); _set(lm, 28, ax + hw, floor_y, vis=0.95)
    _set(lm, 29, ax - 0.02, floor_y + 0.02, vis=0.93)
    _set(lm, 30, ax + hw + 0.02, floor_y + 0.02, vis=0.93)
    _set(lm, 31, ax - 0.03, floor_y + 0.04, vis=0.92)
    _set(lm, 32, ax + hw + 0.03, floor_y + 0.04, vis=0.92)
    # hands on floor ahead of shoulders, elbows bent
    _set(lm, 13, sh_x - sw/2 + rng.uniform(-0.02, 0.02), sh_y + torso*0.4, vis=0.95)
    _set(lm, 14, sh_x + sw/2 + rng.uniform(-0.02, 0.02), sh_y + torso*0.4, vis=0.95)
    _set(lm, 15, sh_x - sw/2 + rng.uniform(0.04, 0.10), floor_y - 0.02, vis=0.93)
    _set(lm, 16, sh_x + sw/2 + rng.uniform(0.04, 0.10), floor_y - 0.02, vis=0.93)
    return lm


_POSE_GENERATORS = {
    "Standing": _standing_pose, "Sitting": _sitting_pose, "Walking": _walking_pose,
    "Running": _running_pose, "Falling": _falling_pose, "Lying Down": _lying_pose,
    "Bending": _bending_pose, "Squatting": _squatting_pose, "Jumping": _jumping_pose,
    "Climbing Stairs": _stairs_pose, "Crouching": _crouching_pose,
    "Kneeling": _kneeling_pose, "Crawling": _crawling_pose,
}


def generate_synthetic_features(budget, seed=42, classes=None):
    """budget: {activity: n_synthetic}. Each synthetic sample is its own group."""
    rng = np.random.RandomState(seed)
    classes = classes or list(ACTIVITIES)
    X_list, y_list, g_list = [], [], []
    for act, n_per_class in budget.items():
        if act not in _POSE_GENERATORS or n_per_class <= 0:
            continue
        gen = _POSE_GENERATORS[act]
        idx = classes.index(act)
        feats = []
        groups = []
        for i in range(n_per_class):
            lm = _build_pose(gen(rng), act)
            f = extract_features(lm)          # synthetic poses are static (no motion)
            if not np.any(np.isnan(f)):
                feats.append(f)
                groups.append(f"synth:{act}:{i}")
        X_list.append(np.array(feats, dtype=np.float32))
        y_list.append(np.full(len(feats), idx, dtype=int))
        g_list.append(np.array(groups))
        print(f"  synthetic {act}: {len(feats)}")
    if not X_list:
        return (np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
                np.empty((0,), dtype=int), np.empty((0,), dtype=object))
    return (np.vstack(X_list), np.concatenate(y_list), np.concatenate(g_list))


def main():
    import logging
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("ExtractFeature")
    PROJECT_DIR = Path(__file__).parent
    DATA_DIR = PROJECT_DIR / "dataset"
    FEATURES_DIR = PROJECT_DIR / "features"
    FEATURES_DIR.mkdir(exist_ok=True)

    real_counts = {}
    real_labels = {}          # activity name -> list of real feature arrays
    real_groups = {}          # activity name -> list of group-id arrays

    # Correct folder-name -> activity mapping
    activity_dirs = [
        ("standing", "Standing"),
        ("sitting", "Sitting"),
        ("bending", "Bending"),
        ("lying", "Lying Down"),
    ]
    for dir_name, act in activity_dirs:
        d = DATA_DIR / dir_name
        if d.exists():
            # These folders are exported frame sequences named 1.jpg,
            # 2.jpg, ... . Keep adjacent frames together during splitting.
            X_c, y_c, g_c, det, total = process_dataset_images(
                d, ACTIVITY_MAP[act], group_strategy="numeric_blocks")
            if len(X_c) > 0:
                real_labels.setdefault(act, []).append(X_c)
                real_groups.setdefault(act, []).append(np.array(g_c))
                real_counts[dir_name] = (det, total)
                logger.info(f"  {dir_name}: {det}/{total} detected -> {act}")

    # fall_dataset: use the YOLO class of the (single) object as the true label.
    fall_img_dir = DATA_DIR / "fall_dataset"
    if fall_img_dir.exists():
        logger.info("Processing fall_dataset (labels decoded from YOLO classes)...")
        for split in ["train", "val"]:
            imgs = fall_img_dir / "images" / split
            lbls = fall_img_dir / "labels" / split
            if not imgs.exists() or not lbls.exists():
                continue
            for label_file in sorted(lbls.glob("*.txt")):
                lines = label_file.read_text().strip().splitlines()
                if len(lines) != 1:      # skip multi-person images (ambiguous label)
                    continue
                cls = int(lines[0].split(" ")[0])
                act = {0: "Falling", 1: "Standing", 2: "Sitting"}[cls]
                img_file = imgs / f"{label_file.stem}.jpg"
                if not img_file.exists():
                    continue
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                landmarks = extract_landmarks(img)
                if landmarks is None:
                    continue
                feats = extract_features(landmarks)
                if not np.any(np.isnan(feats)):
                    real_labels.setdefault(act, []).append(feats.reshape(1, -1))
                    # The YOLO dataset is also an ordered frame export.  Use
                    # short contiguous blocks rather than letting adjacent
                    # frames from one scene land in both train and test.
                    digits = re.findall(r"\d+", img_file.stem)
                    block = (int(digits[-1]) - 1) // 20 if digits else img_file.stem
                    group_id = f"fall_dataset:{split}:{act}:block:{block}"
                    real_groups.setdefault(act, []).append(np.array([group_id]))
                    key = f"fall_dataset/{split}/{cls}:{act}"
                    real_counts[key] = real_counts.get(key, (0, 0))
                    c = real_counts[key]
                    real_counts[key] = (c[0] + 1, c[1] + 1)
            logger.info(f"  fall_dataset/{split} done")

    # Recorded data from the in-app "Record & Train" section.
    # JPG frames are static samples; .avi videos get temporal (motion) features.
    rec_root = DATA_DIR / "recorded"
    if rec_root.exists():
        logger.info("Processing recorded data (Record & Train)...")
        for d in sorted(rec_root.iterdir()):
            if not d.is_dir():
                continue
            act = _ACTIVITY_REMAP.get(d.name, d.name)
            if act not in ACTIVITY_MAP:
                logger.info(f"  skipping unknown activity folder: {act}")
                continue
            # Consecutive webcam frames are not independent samples.  Each
            # timestamp-contiguous capture session is one split group.
            X_c, y_c, g_c, det, total = process_dataset_images(
                d, ACTIVITY_MAP[act], group_strategy="recorded_sessions")
            if len(X_c) > 0:
                real_labels.setdefault(act, []).append(X_c)
                real_groups.setdefault(act, []).append(np.array(g_c))
                real_counts[f"recorded/{act}"] = (det, total)
                logger.info(f"  recorded/{act} frames: {det}/{total} -> {act}")
            X_v, y_v, g_v, det_v = process_hmdb_clips(d, ACTIVITY_MAP[act], max_clips=60)
            if len(X_v) > 0:
                real_labels.setdefault(act, []).append(X_v)
                real_groups.setdefault(act, []).append(np.array(g_v))
                real_counts[f"recorded_videos/{act}"] = (det_v, det_v)
                logger.info(f"  recorded/{act} videos: {det_v} frames -> {act}")

    # UR Fall fall sequences -> real falling frames (with motion features).
    urfall_fall = DATA_DIR / "urfall" / "fall"
    if urfall_fall.exists():
        logger.info("Processing UR Fall fall sequences...")
        X_c, y_c, g_c, det = process_urfall_falls(urfall_fall, ACTIVITY_MAP["Falling"])
        if len(X_c) > 0:
            real_labels.setdefault("Falling", []).append(X_c)
            real_groups.setdefault("Falling", []).append(np.array(g_c))
            real_counts["urfall/fall"] = (det, det)
            logger.info(f"  urfall/fall: {det} real fall frames")

    # IMVIA CCTV fall videos -> real falling frames with motion features.
    imvia_root = DATA_DIR / "imvia"
    if imvia_root.exists():
        logger.info("Processing IMVIA fall videos...")
        X_c, y_c, g_c, det = process_imvia_falls(imvia_root, ACTIVITY_MAP["Falling"])
        if len(X_c) > 0:
            real_labels.setdefault("Falling", []).append(X_c)
            real_groups.setdefault("Falling", []).append(np.array(g_c))
            real_counts["imvia/fall"] = (det, det)
            logger.info(f"  imvia/fall: {det} real fall frames")

    # HMDB51 clips -> real frames for motion activities.
    hmdb_root = DATA_DIR / "hmdb51"
    if hmdb_root.exists():
        logger.info("Processing HMDB51 clips...")
        for d in sorted(hmdb_root.iterdir()):
            if not d.is_dir():
                continue
            act = _ACTIVITY_REMAP.get(d.name, d.name)
            if act not in ACTIVITY_MAP:
                continue
            X_c, y_c, g_c, det = process_hmdb_clips(d, ACTIVITY_MAP[act],
                                                    start_frac=0.4 if act == "Falling" else 0.0)
            if len(X_c) > 0:
                real_labels.setdefault(act, []).append(X_c)
                real_groups.setdefault(act, []).append(np.array(g_c))
                real_counts[f"hmdb51/{act}"] = (det, det)
                logger.info(f"  hmdb51/{act}: {det} real frames")

    real_per_class = {a: sum(len(x) for x in lst) for a, lst in real_labels.items()}
    logger.info(f"Classes with real data: {[a for a, n in real_per_class.items() if n > 0]}")

    # Only train on classes that have enough real data OR a synthetic generator,
    # restricted to the curated KEEP_ACTIVITIES whitelist.
    final_activities = [
        a for a in ACTIVITIES
        if a in KEEP_ACTIVITIES
        and (real_per_class.get(a, 0) >= 4 or a in _POSE_GENERATORS)
    ]
    final_map = {a: i for i, a in enumerate(final_activities)}
    logger.info(f"Final activities ({len(final_activities)}): {final_activities}")

    X_list, y_list, g_list = [], [], []
    for act in final_activities:
        if act in real_labels:
            for X_c, g_c in zip(real_labels[act], real_groups.get(act, [])):
                X_list.append(X_c)
                y_list.append(np.full(len(X_c), final_map[act], dtype=int))
                g_list.append(np.asarray(g_c, dtype=object))
    X_real = np.vstack(X_list) if X_list else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    y_real = np.concatenate(y_list) if y_list else np.empty((0,), dtype=int)
    g_real = np.concatenate(g_list) if g_list else np.empty((0,), dtype=object)
    logger.info(f"Total real samples: {len(X_real)}")

    # Synthetic supplement only where real data is thin.
    synth_budget = {a: _synthetic_budget(a, real_per_class.get(a, 0)) for a in final_activities}
    synth_budget = {a: n for a, n in synth_budget.items() if n > 0}
    if synth_budget:
        logger.info(f"Generating synthetic data to supplement: {synth_budget}")
        X_synth, y_synth, g_synth = generate_synthetic_features(synth_budget, seed=42, classes=final_activities)
    else:
        X_synth = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        y_synth = np.empty((0,), dtype=int)
        g_synth = np.empty((0,), dtype=object)

    X = np.vstack([X_real, X_synth])
    y = np.concatenate([y_real, y_synth])
    groups = np.concatenate([g_real, g_synth])
    logger.info(f"Total samples: {len(X)}")
    for i, act in enumerate(final_activities):
        logger.info(f"  {act}: {int(np.sum(y == i))}")

    # Group-aware split: frames from the same clip/sequence never cross splits,
    # so test accuracy reflects truly unseen footage.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_train0, X_test, y_train0, y_test = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
    g_train0, g_test = groups[tr_idx], groups[te_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=42)
    tr2, va_idx = next(gss2.split(X_train0, y_train0, g_train0))
    X_train, X_val, y_train, y_val = X_train0[tr2], X_train0[va_idx], y_train0[tr2], y_train0[va_idx]
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)} (group split)")

    np.savez(str(FEATURES_DIR / "features.npz"),
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    logger.info(f"Features saved to {FEATURES_DIR / 'features.npz'}")

    info = {"activities": final_activities, "feature_names": FEATURE_NAMES, "real_counts": real_counts}
    with open(FEATURES_DIR / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Done!")


if __name__ == "__main__":
    main()

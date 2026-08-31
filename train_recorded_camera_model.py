"""Train a camera-snapshot activity model from dataset/recorded.

This model intentionally learns the camera angle used by the in-app recording
station.  It is used only as a static-photo activity aid; the main full-body
model remains responsible for live monitoring and fall alerts.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from extract_feature import extract_landmarks
from features import FEATURE_NAMES, extract_features


PROJECT_DIR = Path(__file__).parent
RECORDED_DIR = PROJECT_DIR / "dataset" / "recorded"
OUTPUT_PATH = PROJECT_DIR / "models" / "recorded_camera_model.pkl"
REMAP = {"Sitting on Floor": "Sitting"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    samples, labels = [], []
    for directory in sorted(RECORDED_DIR.iterdir()):
        if not directory.is_dir():
            continue
        label = REMAP.get(directory.name, directory.name)
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            landmarks = extract_landmarks(image)
            if landmarks is None:
                continue
            features = extract_features(landmarks)
            if not np.any(np.isnan(features)):
                samples.append(features)
                labels.append(label)

    if len(samples) < 20 or len(set(labels)) < 2:
        raise RuntimeError("Not enough detected recorded frames to train a camera model.")

    X = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels)
    model = RandomForestClassifier(
        n_estimators=400, max_depth=18, min_samples_leaf=1,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    model.fit(X, y)

    # This is a sanity check, not a generalization claim: adjacent recording
    # frames are deliberately retained because this model calibrates the same
    # webcam setup used for the snapshot fallback.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Recorded camera samples: {len(X)}")
    print("Class counts:", {label: int(np.sum(y == label)) for label in sorted(set(y))})
    print(f"Holdout accuracy: {model.score(X_test, y_test):.3f}")

    joblib.dump({
        "model": model,
        "feature_names": FEATURE_NAMES,
        "classes": list(model.classes_),
        "sample_count": len(X),
    }, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

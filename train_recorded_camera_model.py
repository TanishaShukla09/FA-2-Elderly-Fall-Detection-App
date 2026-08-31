"""Train a camera-snapshot activity model from the full labelled image dataset.

This model learns the camera-angle appearance used by the in-app recording
station plus all per-activity image datasets (sitting, standing, lying,
bending) so a captured webcam photo can be classified reliably.  It is used
only as a static-photo activity aid; the main full-body model remains
responsible for live monitoring and fall alerts.
"""
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from extract_feature import extract_landmarks
from features import FEATURE_NAMES, extract_features


PROJECT_DIR = Path(__file__).parent
OUTPUT_PATH = PROJECT_DIR / "models" / "recorded_camera_model.pkl"
REMAP = {"Sitting on Floor": "Sitting", "Sitting Down": "Sitting"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
# subdirectories -> activity label, for the recorded/ and any labelled folders
# whose sub-folder names are already activity names.
LABELLED_SUBDIRS = {
    "recorded": {
        "Crouching": "Crouching", "Kneeling": "Kneeling",
        "Lying Down": "Lying Down", "Sitting": "Sitting",
        "Sitting on Floor": "Sitting", "Standing": "Standing",
        "Walking": "Walking",
    },
}
# Flat folders whose name is the activity label.
FLAT_LABELED = {
    "sitting": "Sitting",
    "standing": "Standing",
    "lying": "Lying Down",
}


def _collect(folder, label):
    """Extract features+labels from every image in `folder` under `label`."""
    samples, labels = [], []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        landmarks = extract_landmarks(image)
        if landmarks is None:
            continue
        features = extract_features(landmarks)
        if np.any(np.isnan(features)):
            continue
        samples.append(features)
        labels.append(label)
    return samples, labels


def main():
    samples, labels = [], []

    # Per-activity flat image folders (dataset/sitting, ...).
    for subdir, label in FLAT_LABELED.items():
        folder = PROJECT_DIR / "dataset" / subdir
        if not folder.is_dir():
            continue
        s, l = _collect(folder, label)
        samples.extend(s); labels.extend(l)
        print(f"{subdir:12s} -> {label:12s}: {len(s)} frames")

    # Recorded sub-directory folders (dataset/recorded/<Activity>).
    recorded_dir = PROJECT_DIR / "dataset" / "recorded"
    label_map = LABELLED_SUBDIRS.get("recorded", {})
    for directory in sorted(recorded_dir.iterdir()):
        if not directory.is_dir():
            continue
        label = label_map.get(directory.name, REMAP.get(directory.name, directory.name))
        s, l = _collect(directory, label)
        samples.extend(s); labels.extend(l)
        print(f"recorded/{directory.name:12s} -> {label:12s}: {len(s)} frames")

    if len(samples) < 20 or len(set(labels)) < 2:
        raise RuntimeError("Not enough detected frames to train a camera model.")

    X = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels)
    # HistGradientBoosting gives near-RF accuracy on tabular joint features but a
    # model file ~100x smaller than a RandomForest (RF on thousands of trees can
    # exceed GitHub's 100 MB limit).
    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=0.1, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25,
        class_weight="balanced", random_state=42,
    )
    model.fit(X, y)

    # Sanity check only: this model calibrates the app's webcam snapshot aid
    # (same camera angle), so adjacent/recording frames are deliberately kept.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Total camera samples: {len(X)}")
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

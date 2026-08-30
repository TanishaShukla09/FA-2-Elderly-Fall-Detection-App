"""
Train a Keras MLP fall-detection model on the IMVIA pose features and
auto-generate all requested artifacts:

    models/fall detection.h5        Keras model (Standing / Falling / Lying Down)
    models/history.pkl              training history
    models/label_encoder.pkl        LabelEncoder (class name <-> index)
    models/threshold.pkl            best F1 fall-alarm threshold on the Falling class
    models/x_Test.npy  models/ytest.npy    held-out test set
    screenshots/training_curves.png         accuracy/loss vs epoch
    screenshots/confusion_matrix_nn.png
    screenshots/classification_report_nn.txt / _nn.png

Requires: python extract_imvia.py first.
"""
import argparse
import json
import os
import pickle

import numpy as np
import joblib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow import keras
from tensorflow.keras import layers

logging_fmt = "%(asctime)s [%(levelname)s] %(message)s"
import logging
logging.basicConfig(level=logging.INFO, format=logging_fmt)
logger = logging.getLogger("TrainNN")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
SCREENSHOT_DIR = os.path.join(PROJECT_DIR, "screenshots")
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)


def load_imvia_features():
    npz = os.path.join(FEATURES_DIR, "imvia_features.npz")
    info_json = os.path.join(FEATURES_DIR, "imvia_info.json")
    if not os.path.exists(npz):
        logger.error(f"Missing {npz} - run 'python extract_imvia.py' first.")
        return None
    data = np.load(npz)
    with open(info_json) as f:
        info = json.load(f)
    return data["X"], data["y"], info["class_names"], info["feature_names"]


def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Training Curves (Keras MLP)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved -> {save_path}")


def plot_confusion(cm, classes, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Keras MLP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved -> {save_path}")


def plot_report(df, save_path):
    plt.figure(figsize=(8, 1.2 + 0.55 * len(df)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu",
                cbar=False, annot_kws={"size": 10})
    plt.title("Classification Report (Keras MLP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved -> {save_path}")


def best_threshold(y_true_binary, prob):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.01, 0.99, 0.01):
        f1 = f1_score(y_true_binary, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--binary", action="store_true",
                        help="Train 2-class (Falling vs Not-Falling) instead of 3-class")
    args = parser.parse_args()

    loaded = load_imvia_features()
    if loaded is None:
        return
    X, y, class_names, feature_names = loaded
    logger.info(f"Loaded {len(X)} samples, {X.shape[1]} features, classes={class_names}")

    le = LabelEncoder()
    if args.binary:
        fall_idx = class_names.index("Falling")
        y = (y == fall_idx).astype(int)
        class_names = ["Not Falling", "Falling"]
        le.fit(class_names)
    else:
        le.fit(class_names)
    # Keep model-output order (index i <-> class_names[i]) instead of alphabetical.
    le.classes_ = np.asarray(class_names)
    label_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    joblib.dump(le, label_path)
    logger.info(f"Saved -> {label_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15 / 0.85, random_state=SEED, stratify=y_train)
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    n_classes = len(class_names)
    norm = layers.Normalization()
    norm.adapt(X_train)

    model = keras.Sequential([
        keras.Input(shape=(X.shape[1],)),
        norm,
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=30,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=10, min_lr=1e-6, verbose=1),
    ]
    counts = np.bincount(y_train)
    class_weight = {i: len(y_train) / (len(counts) * c) for i, c in enumerate(counts) if c > 0}
    logger.info(f"Class weights: {class_weight}")
    logger.info("Training Keras MLP...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=args.epochs, batch_size=args.batch_size,
                        callbacks=callbacks, class_weight=class_weight, verbose=2)

    hist_path = os.path.join(MODEL_DIR, "history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(dict(history.history), f)
    logger.info(f"Saved -> {hist_path}")

    h5_path = os.path.join(MODEL_DIR, "fall detection.h5")
    model.save(h5_path)
    logger.info(f"Saved -> {h5_path}")

    plot_curves(history.history, os.path.join(SCREENSHOT_DIR, "training_curves.png"))

    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    logger.info(f"Test  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names,
                                   zero_division=0)
    logger.info("\n" + report)
    report_path = os.path.join(SCREENSHOT_DIR, "classification_report_nn.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved -> {report_path}")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion(cm, class_names, os.path.join(SCREENSHOT_DIR, "confusion_matrix_nn.png"))

    report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                        zero_division=0, output_dict=True)
    import pandas as pd
    rows = {k: v for k, v in report_dict.items() if k not in ("accuracy",)}
    df = pd.DataFrame(rows).T.drop(columns=["support"])
    plot_report(df, os.path.join(SCREENSHOT_DIR, "classification_report_nn.png"))

    np.save(os.path.join(MODEL_DIR, "x_Test.npy"), X_test)
    np.save(os.path.join(MODEL_DIR, "ytest.npy"), y_test)
    logger.info(f"Saved -> {MODEL_DIR}/x_Test.npy, ytest.npy")

    if not args.binary and "Falling" in class_names:
        fall_idx = class_names.index("Falling")
        val_probs = model.predict(X_val, verbose=0)[:, fall_idx]
        val_bin = (y_val == fall_idx).astype(int)
        t, best_f1 = best_threshold(val_bin, val_probs)
        threshold = {"threshold": t, "fall_class": "Falling",
                     "classes": class_names, "f1_at_threshold": best_f1,
                     "note": "Alarm when P(Falling) >= threshold"}
    else:
        threshold = {"threshold": 0.5, "fall_class": "Falling",
                     "classes": class_names, "f1_at_threshold": None}
    thr_path = os.path.join(MODEL_DIR, "threshold.pkl")
    with open(thr_path, "wb") as f:
        pickle.dump(threshold, f)
    logger.info(f"Saved -> {thr_path} | {threshold}")

    fall_idx = class_names.index("Falling")
    y_bin = (y_test == fall_idx).astype(int)
    prob_fall = probs[:, fall_idx]
    t_val = threshold["threshold"]
    pred_bin = (prob_fall >= t_val).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()
    bin_acc = accuracy_score(y_bin, pred_bin)
    bin_prec = precision_score(y_bin, pred_bin, zero_division=0)
    bin_rec = recall_score(y_bin, pred_bin, zero_division=0)
    bin_f1 = f1_score(y_bin, pred_bin, zero_division=0)
    logger.info(f"Fall vs Not-Fall @ t={t_val:.2f}: TP={tp} FP={fp} TN={tn} FN={fn}")
    logger.info(f"  Accuracy: {bin_acc:.4f} | Precision: {bin_prec:.4f} | "
                f"Recall: {bin_rec:.4f} | F1: {bin_f1:.4f}")
    cm_bin = np.array([[tn, fp], [fn, tp]])
    plot_confusion(cm_bin, ["Not Fall", "Fall"],
                   os.path.join(SCREENSHOT_DIR, "confusion_matrix_fall_not_fall.png"))

    logger.info("=" * 60)
    logger.info("KERAS TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

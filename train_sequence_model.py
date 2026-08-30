"""Train an LSTM (or GRU) binary fall detector on 30-frame feature sequences.

Compares sequence-level fall metrics (precision / recall / F1 / F2) against the
classical frame-level binary fall model evaluated on the SAME held-out test
clips. Sequence decision for the frame model = max frame fall-probability over
the window (a fall is called if any frame in the window alarms).

Usage:  python train_sequence_model.py [gru|lstm]
Output: models/fall_sequence_model.keras + screenshots/sequence_model_report.txt
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import joblib

import config
from features import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainSequence")

MODEL_PATH = config.SEQUENCE_MODEL_PATH


def load():
    data = np.load(config.FEATURES_DIR / "sequences.npz", allow_pickle=True)
    info = json.loads((config.FEATURES_DIR / "sequences_info.json").read_text())
    return data, info


def normalize(X, mean, std):
    return (X - mean) / std


def build_model(seq_len, input_dim, cell="gru", seed=42):
    import keras
    from keras import layers
    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(seq_len, input_dim))
    unit = layers.GRU if cell == "gru" else layers.LSTM
    x = unit(128, return_sequences=True)(inp)
    x = layers.Dropout(0.3)(x)
    x = unit(64)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, x)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fall_metrics(y_true, y_pred_prob, threshold):
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                 fbeta_score, accuracy_score)
    yb = (y_pred_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, yb)),
        "precision": float(precision_score(y_true, yb, zero_division=0)),
        "recall": float(recall_score(y_true, yb, zero_division=0)),
        "f1": float(f1_score(y_true, yb, zero_division=0)),
        "f2": float(fbeta_score(y_true, yb, beta=2.0, zero_division=0)),
        "tp": int(np.sum((yb == 1) & (y_true == 1))),
        "fp": int(np.sum((yb == 1) & (y_true == 0))),
        "tn": int(np.sum((yb == 0) & (y_true == 0))),
        "fn": int(np.sum((yb == 0) & (y_true == 1))),
    }


def tune_threshold(y_true, y_prob):
    from sklearn.metrics import fbeta_score
    best_t, best_f2 = 0.55, -1.0
    for t in np.arange(0.10, 0.95, 0.01):
        f2 = fbeta_score(y_true, (y_prob >= t).astype(int), beta=2.0, zero_division=0)
        if f2 > best_f2:
            best_t, best_f2 = float(t), float(f2)
    return best_t, float(best_f2)


def frame_model_sequence_predictions(model, X, normalize_fun):
    """Per-sequence decision for the classical model: max frame probability."""
    n = len(X)
    flat = X.reshape(-1, X.shape[-1])
    p = model.predict_proba(normalize_fun(flat))[:, 1]
    p = p.reshape(n, -1)
    return p.max(axis=1)


def main():
    cell = sys.argv[1].lower() if len(sys.argv) > 1 else "gru"
    logger.info("=" * 60)
    logger.info(f"Sequence fall detector ({cell.upper()})")
    logger.info("=" * 60)

    data, info = load()
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"], data["y_val"]
    X_te, y_te = data["X_test"], data["y_test"]
    seq_len = info["seq_len"]
    logger.info(f"Sequences: train {len(X_tr)} | val {len(X_va)} | test {len(X_te)}")
    logger.info(f"Fall share: train {y_tr.mean():.3f} val {y_va.mean():.3f} test {y_te.mean():.3f}")

    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-6
    nX = lambda x: normalize(x, mean, std)

    model = build_model(seq_len, X_tr.shape[-1], cell=cell)
    model.summary(print_fn=logger.info)

    # Emphasise falls in the loss; the F2 threshold tuning below then sets the
    # alarm operating point (recall-weighted) on the validation set.
    class_weight = {0: 1.0, 1: 1.5}

    from keras import callbacks as cb
    cbs = [
        cb.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    logger.info(f"Training {cell.upper()} with class_weight={class_weight}...")
    model.fit(nX(X_tr), y_tr, validation_data=(nX(X_va), y_va),
              epochs=60, batch_size=512, callbacks=cbs, class_weight=class_weight, verbose=1)

    val_prob = model.predict(nX(X_va), verbose=0).ravel()
    thr, val_f2 = tune_threshold(y_va, val_prob)
    logger.info(f"Tuned alarm threshold (val F2): {thr:.2f} (F2={val_f2:.3f})")

    test_prob = model.predict(nX(X_te), verbose=0).ravel()
    lstm_metrics = fall_metrics(y_te, test_prob, thr)
    logger.info(f"[{cell.upper()}] test fall metrics: {lstm_metrics}")

    # ---- Classical frame-model comparison on the same test sequences ----
    frame_metrics = None
    frame_path = config.MODEL_DIR / "binary_fall_model.pkl"
    if frame_path.exists():
        bundle = joblib.load(frame_path)
        fm = bundle["model"]
        seq_pred = frame_model_sequence_predictions(fm, X_te, lambda x: x)
        fthr = bundle.get("threshold", 0.55)
        frame_metrics = fall_metrics(y_te, seq_pred, fthr)
        logger.info(f"[frame model] test fall metrics @ {fthr:.2f}: {frame_metrics}")

    model.save(MODEL_PATH)
    logger.info(f"Saved -> {MODEL_PATH}")

    (config.MODEL_DIR / "sequence_normalizer.json").write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist()}))
    logger.info(f"Saved -> {config.MODEL_DIR / 'sequence_normalizer.json'}")

    report = []
    report.append("=" * 60)
    report.append("Sequence vs Frame-level fall detector (held-out test clips)")
    report.append("=" * 60)
    for name, m in [("Frame-level (classical)", frame_metrics), (f"{cell.upper()} (sequence)", lstm_metrics)]:
        report.append(f"\n--- {name} ---")
        if m:
            for k, v in m.items():
                report.append(f"  {k}: {v}")
    (config.SCREENSHOT_DIR / "sequence_model_report.txt").write_text("\n".join(report))
    logger.info(f"Saved -> {config.SCREENSHOT_DIR / 'sequence_model_report.txt'}")


if __name__ == "__main__":
    main()

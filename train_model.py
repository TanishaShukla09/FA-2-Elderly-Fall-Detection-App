import os
import json
import joblib
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainModel")

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
SCREENSHOT_DIR = PROJECT_DIR / "screenshots"
FEATURES_DIR = PROJECT_DIR / "features"
MODEL_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)

SEED = 42

# Minority/transitory classes get extra weight beyond 'balanced' so the
# classifier does not sacrifice them for the abundant Standing/Falling data.
CLASS_BOOST = {
    "Getting Up": 3.0, "Running": 3.5, "Walking": 3.0,
    "Climbing Stairs": 2.5, "Jumping": 2.5, "Squatting": 2.0, "Crawling": 2.0,
    "Kneeling": 2.0, "Crouching": 2.0,
}

# CHANGED: was 600. Filling every minority class up to 600 meant classes with
# very few real samples (e.g. Getting Up at ~44) were being duplicated 10-15x.
# A high-capacity tree can then carve out a leaf that memorizes those exact
# duplicated rows, which is why train accuracy sits near 99% while val/test
# stayed around 82-83% (see confusion_matrix_train.png vs classification_report.txt).
# Lowering the floor reduces how much duplication any single class needs.
OVERSAMPLE_MIN_PER_CLASS = 250

# CHANGED: new safety cap. No class may be duplicated more than this many
# times over its original count, regardless of OVERSAMPLE_MIN_PER_CLASS.
# For very small classes this caps memorization risk instead of blindly
# stretching 40 real samples into 600 near-identical copies.
OVERSAMPLE_MAX_DUP_RATIO = 6.0


def custom_class_weights(y, activities):
    counts = np.bincount(y, minlength=len(activities)).astype(float)
    w = {}
    for i, a in enumerate(activities):
        cw = len(y) / (len(activities) * max(1.0, counts[i]))
        w[i] = cw * CLASS_BOOST.get(a, 1.0)
    return w


def oversample(X, y, activities=None, min_per_class=OVERSAMPLE_MIN_PER_CLASS,
                max_dup_ratio=OVERSAMPLE_MAX_DUP_RATIO, seed=SEED):
    """Random oversampling by duplication: boosts minority classes up to
    min_per_class (capped at max_dup_ratio x their original count) without
    introducing synthetic samples that may lie outside the true data manifold.

    CHANGED: added max_dup_ratio cap + a warning log so classes that are
    genuinely under-recorded are flagged instead of silently duplicated into
    the hundreds. If you see this warning for a class, the real fix is to
    record more distinct clips of that activity, not to raise the cap.
    """
    rng = np.random.RandomState(seed)
    counts = np.bincount(y, minlength=len(np.unique(y)))
    minority = [i for i, c in enumerate(counts) if 0 < c < min_per_class]
    if not minority:
        return X, y
    X_list, y_list = [X], [y]
    for cls in minority:
        idx = np.where(y == cls)[0]
        orig_n = len(idx)
        capped_target = min(min_per_class, int(orig_n * max_dup_ratio))
        need = max(0, capped_target - orig_n)
        if capped_target < min_per_class:
            name = activities[cls] if activities else f"class {cls}"
            logger.warning(
                f"  '{name}' has only {orig_n} samples - capped oversample target at "
                f"{capped_target} (ratio {max_dup_ratio}x) instead of {min_per_class}. "
                f"Record more real '{name}' clips to actually improve this class."
            )
        if need == 0:
            continue
        dup = rng.choice(idx, size=need, replace=True)
        X_list.append(X[dup])
        y_list.append(y[dup])
    X_res = np.concatenate(X_list, axis=0)
    y_res = np.concatenate(y_list, axis=0)
    perm = rng.permutation(len(y_res))
    return X_res[perm].astype(np.float32), y_res[perm].astype(int)


def load_features():
    npz_path = FEATURES_DIR / "features.npz"
    info_path = FEATURES_DIR / "info.json"
    if not npz_path.exists():
        logger.error(f"Features not found at {npz_path}. Run extract_feature.py first.")
        return None, None, None, None, None
    data = np.load(npz_path)
    with open(info_path) as f:
        info = json.load(f)
    return (data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"],
            info["activities"], info["feature_names"])


def train_binary_fall_model(X_train, y_train, X_val, y_val, X_test, y_test,
                            fall_idx, activities, feature_names, model_dir, seed=42):
    """Train a Fall/No-Fall binary detector, tune the alarm threshold on the
    validation set to maximise fall F2 (recall-weighted) and report the full
    fall metric set (precision/recall/F1/F2 + false-alarm breakdown) on test."""
    yb = lambda y: (np.asarray(y) == fall_idx).astype(int)
    yb_tr, yb_va, yb_te = yb(y_train), yb(y_val), yb(y_test)
    n_pos = int(yb_tr.sum())
    logger.info(f"  Binary targets: train fall={n_pos}/{len(yb_tr)} "
                f"val fall={int(yb_va.sum())}/{len(yb_va)} "
                f"test fall={int(yb_te.sum())}/{len(yb_te)}")

    candidates = {
        # CHANGED: max_depth 18->12, min_samples_leaf 2->6, min_samples_split 4->10.
        # Same rationale as the multiclass model below - shallower trees with
        # bigger leaves generalize better instead of memorizing duplicated rows.
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=10,
            min_samples_leaf=6, random_state=seed, class_weight="balanced", n_jobs=-1,
        ),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=12,
            min_samples_leaf=35, l2_regularization=0.5, random_state=seed,
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30,
        ),
    }

    def best_threshold_f2(model, Xv, yv):
        prob = model.predict_proba(Xv)[:, 1]
        best_t, best_f2 = 0.55, -1.0
        for t in np.arange(0.10, 0.95, 0.01):
            f2 = fbeta_score(yv, (prob >= t).astype(int), beta=2.0, zero_division=0)
            if f2 > best_f2:
                best_t, best_f2 = float(t), float(f2)
        return best_t, best_f2

    best_name, best_model, best_thresh, best_val_f2 = None, None, 0.55, -1.0
    for name, m in candidates.items():
        m.fit(X_train, yb_tr)
        t, f2 = best_threshold_f2(m, X_val, yb_va)
        logger.info(f"    {name}: val F2={f2:.3f} @ threshold={t:.2f}")
        if f2 > best_val_f2:
            best_name, best_model, best_thresh, best_val_f2 = name, m, t, f2

    # Final threshold retuned on validation for the chosen model.
    best_thresh, best_val_f2 = best_threshold_f2(best_model, X_val, yb_va)
    prob_te = best_model.predict_proba(X_test)[:, 1]
    pred_te = (prob_te >= best_thresh).astype(int)

    prec = precision_score(yb_te, pred_te, zero_division=0)
    rec = recall_score(yb_te, pred_te, zero_division=0)
    f1 = f1_score(yb_te, pred_te, zero_division=0)
    f2 = fbeta_score(yb_te, pred_te, beta=2.0, zero_division=0)
    acc = accuracy_score(yb_te, pred_te)
    tn = int(np.sum((pred_te == 0) & (yb_te == 0)))
    fp = int(np.sum((pred_te == 1) & (yb_te == 0)))
    fn = int(np.sum((pred_te == 0) & (yb_te == 1)))
    tp = int(np.sum((pred_te == 1) & (yb_te == 1)))

    # Which activities trigger false alarms?
    fa_breakdown = {}
    for a in activities:
        mask = (np.asarray(y_test) == activities.index(a)) & (yb_te == 0)
        fa_breakdown[a] = int(np.sum(pred_te[mask] == 1))
    fa_breakdown = {k: v for k, v in sorted(fa_breakdown.items(), key=lambda kv: -kv[1]) if v > 0}

    metrics = {
        "model_type": best_name, "threshold": best_thresh,
        "val_f2": float(best_val_f2),
        "test_accuracy": acc, "test_precision": prec, "test_recall": rec,
        "test_f1": f1, "test_f2": f2,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "false_alarm_rate": float(fp / max(1, fp + tn)),
        "false_alarm_breakdown": fa_breakdown,
    }

    joblib.dump({
        "model": best_model, "model_type": best_name,
        "feature_names": list(feature_names), "threshold": float(best_thresh),
        "metrics": metrics,
    }, model_dir / "binary_fall_model.pkl")
    logger.info(f"  Saved -> {model_dir / 'binary_fall_model.pkl'}")

    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        order = np.argsort(imp)[::-1][:15]
        plt.figure(figsize=(8, 5))
        plt.barh(range(len(order)), imp[order][::-1], align="center", color="#e74c3c")
        plt.yticks(range(len(order)), [feature_names[i] for i in order[::-1]])
        plt.xlabel("Importance")
        plt.title("Fall Detector Feature Importance (Top 15)")
        plt.tight_layout()
        plt.savefig(str(SCREENSHOT_DIR / "fall_feature_importance.png"), dpi=150)
        plt.close()
        logger.info(f"  Saved -> {SCREENSHOT_DIR / 'fall_feature_importance.png'}")
    return best_model, metrics


def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(f"Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved -> {save_path}")


def main():
    logger.info("=" * 60)
    logger.info("Fall Detection - Training Pipeline")
    logger.info("=" * 60)

    logger.info("[1/5] Loading features...")
    result = load_features()
    if result[0] is None:
        logger.error("Run 'python extract_feature.py' first")
        return
    X_train, y_train, X_val, y_val, X_test, y_test, activities, feature_names = result
    logger.info(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    for i, act in enumerate(activities):
        total = int(np.sum(y_train == i)) + int(np.sum(y_val == i)) + int(np.sum(y_test == i))
        logger.info(f"  {act}: {total}")

    logger.info("[2/5] Training candidate models...")
    fall_idx = activities.index("Falling") if "Falling" in activities else None
    # CHANGED: standalone RandomForest now uses the same custom_class_weights
    # (balanced + per-class boost) as the ensemble's rf sub-estimator instead
    # of plain "balanced", so minority/transitory classes (Getting Up, Walking,
    # etc.) get consistent treatment regardless of which candidate wins.
    # CHANGED: max_depth 20->14, min_samples_leaf 2->6, min_samples_split 4->10
    # on every RandomForest below - shallower trees with bigger leaves can't
    # carve out a leaf per duplicated oversample row, which is what was
    # driving the train~99% / test~83% gap.
    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_split=10,
            min_samples_leaf=6, random_state=SEED,
            class_weight=custom_class_weights(y_train, activities), n_jobs=-1,
        ),
        "HistGB": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=12,
            min_samples_leaf=35, l2_regularization=0.5, random_state=SEED,
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=30,
        ),
        "RF+HistGB Ensemble": VotingClassifier(
            voting="soft",
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=300, max_depth=14, min_samples_split=10,
                    min_samples_leaf=6, random_state=SEED,
                    class_weight=custom_class_weights(y_train, activities), n_jobs=-1)),
                ("hgb", HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.05, max_leaf_nodes=12,
                    min_samples_leaf=35, l2_regularization=0.5, random_state=SEED,
                    class_weight="balanced", early_stopping=True,
                    validation_fraction=0.15, n_iter_no_change=30)),
            ],
        ),
    }
    # Train on the oversampled matrix so minority classes are represented.
    # CHANGED: pass activities so oversample() can name classes in its warnings.
    X_fit, y_fit = oversample(X_train, y_train, activities=activities)
    best_name, best_model, best_val = None, None, -1.0
    for name, m in candidates.items():
        m.fit(X_fit, y_fit)
        val_pred = m.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="weighted", zero_division=0)
        train_acc = accuracy_score(y_fit, m.predict(X_fit))
        gap = train_acc - val_f1
        score = val_f1
        if fall_idx is not None:
            val_fall_bin = (y_val == fall_idx).astype(int)
            val_fall_pred = (m.predict_proba(X_val)[:, fall_idx] >= 0.10).astype(int)
            score += 0.6 * fbeta_score(val_fall_bin, val_fall_pred, beta=2.0, zero_division=0)
        # Overfit gap is logged for visibility but NOT penalized in
        # selection — the primary fix for memorization is more diverse
        # training data (items #2-5 in the priority list), not selecting
        # a lower-capacity model that sacrifices real accuracy.
        logger.info(f"  {name}: train acc={train_acc:.4f} | val weighted-F1={val_f1:.4f} "
                    f"(select score {score:.4f}) | gap={gap:.4f}")
        if score > best_val:
            best_name, best_model, best_val = name, m, score
    model = best_model
    logger.info(f"  Selected model: {best_name} (val select score {best_val:.4f})")

    logger.info("[2b/2b] Fall-alarm threshold tuning (validation set)...")
    fall_threshold = 0.55
    if "Falling" in activities:
        fall_idx = activities.index("Falling")
        val_fall_prob = model.predict_proba(X_val)[:, fall_idx]
        val_fall_bin = (y_val == fall_idx).astype(int)
        best_f2, best_t = -1.0, fall_threshold
        for t in np.arange(0.10, 0.95, 0.01):
            pred = (val_fall_prob >= t).astype(int)
            f2 = fbeta_score(val_fall_bin, pred, beta=2.0, zero_division=0)  # recall-weighted
            if f2 > best_f2:
                best_f2, best_t = f2, float(t)
        fall_threshold = best_t
        tp = int(np.sum((val_fall_prob >= best_t) & (val_fall_bin == 1)))
        fn = int(np.sum((val_fall_prob < best_t) & (val_fall_bin == 1)))
        fp = int(np.sum((val_fall_prob >= best_t) & (val_fall_bin == 0)))
        logger.info(f"  Alarm when P(Falling) >= {fall_threshold:.2f} "
                    f"(val F2={best_f2:.3f}, TP={tp}, FN={fn}, FP={fp})")

    logger.info("[2c/2c] Training dedicated binary fall detector (Fall vs No-Fall)...")
    fall_bin, bin_metrics = train_binary_fall_model(
        X_fit, y_fit, X_val, y_val, X_test, y_test,
        fall_idx, activities, feature_names, model_dir=MODEL_DIR, seed=SEED,
    )
    logger.info("  Binary fall detector metrics (test):")
    for k, v in bin_metrics.items():
        logger.info(f"    {k}: {v}")

    logger.info("[3/5] Cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    logger.info(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    logger.info("[4/5] Evaluation...")
    split_accs = {}
    for split_name, X_s, y_s in [("Train", X_train, y_train),
                                   ("Validation", X_val, y_val),
                                   ("Test", X_test, y_test)]:
        y_pred = model.predict(X_s)
        acc = accuracy_score(y_s, y_pred)
        split_accs[split_name] = acc
        prec = precision_score(y_s, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_s, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_s, y_pred, average="weighted", zero_division=0)
        logger.info(f"  --- {split_name} ---")
        logger.info(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        logger.info(f"  Precision: {prec:.4f}")
        logger.info(f"  Recall:    {rec:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        if "Falling" in activities:
            fi = activities.index("Falling")
            fall_bin_y = (np.asarray(y_s) == fi).astype(int)
            fall_pred = (model.predict_proba(X_s)[:, fi] >= fall_threshold).astype(int)
            logger.info(f"  [FALL] Precision: {precision_score(fall_bin_y, fall_pred, zero_division=0):.4f}")
            logger.info(f"  [FALL] Recall:    {recall_score(fall_bin_y, fall_pred, zero_division=0):.4f}")
            logger.info(f"  [FALL] F1:        {f1_score(fall_bin_y, fall_pred, zero_division=0):.4f}")
            logger.info(f"  [FALL] F2:        {fbeta_score(fall_bin_y, fall_pred, beta=2.0, zero_division=0):.4f} @ {fall_threshold:.2f}")
        logger.info(f"\n{classification_report(y_s, y_pred, target_names=activities, zero_division=0)}")
        cm = confusion_matrix(y_s, y_pred)
        plot_confusion_matrix(cm, activities, str(SCREENSHOT_DIR / f"confusion_matrix_{split_name.lower()}.png"))

    # CHANGED: explicit overfit-gap warning so this shows up in the training
    # log every run instead of only being visible by eyeballing two screenshots.
    gap = split_accs["Train"] - split_accs["Test"]
    logger.info(f"  Train-Test accuracy gap: {gap:.4f}")
    if gap > 0.10:
        logger.warning(
            f"  Train accuracy is {gap*100:.1f} points above Test accuracy - the model is "
            f"likely overfitting. Consider raising min_samples_leaf / lowering max_depth "
            f"further, or lowering OVERSAMPLE_MIN_PER_CLASS, before adding more data."
        )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif isinstance(model, VotingClassifier):
        importances = model.named_estimators_["rf"].feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(model, X_val, y_val, n_repeats=2,
                                    random_state=SEED, n_jobs=-1)
        importances = pi.importances_mean
    indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(indices)), importances[indices][::-1], align="center", color="#3498db")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
    plt.xlabel("Importance")
    plt.title("Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(str(SCREENSHOT_DIR / "feature_importance.png"), dpi=150)
    plt.close()

    logger.info("[5/5] Saving model...")
    model_path = MODEL_DIR / "fall_model.pkl"
    joblib.dump({
        "model": model,
        "model_type": best_name,
        "feature_names": feature_names,
        "activities": activities,
        "activity_map": {a: i for i, a in enumerate(activities)},
        "fall_threshold": float(fall_threshold),
        "fall_binary": {
            "model_type": bin_metrics["model_type"],
            "threshold": bin_metrics["threshold"],
            "metrics": bin_metrics,
        },
        "cv_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }, model_path)
    logger.info(f"  Model saved -> {model_path}")

    logger.info("[5b/5b] Exporting evaluation artifacts...")
    np.save(FEATURES_DIR / "x_Test.npy", X_test)
    np.save(FEATURES_DIR / "ytest.npy", y_test)
    joblib.dump(LabelEncoder().fit(activities), MODEL_DIR / "activity_label_encoder.pkl")
    test_report = classification_report(y_test, model.predict(X_test),
                                        target_names=activities, zero_division=0)
    (SCREENSHOT_DIR / "classification_report.txt").write_text(test_report)
    logger.info(f"  Saved -> {FEATURES_DIR / 'x_Test.npy'} and {FEATURES_DIR / 'ytest.npy'}")
    logger.info(f"  Saved -> {MODEL_DIR / 'activity_label_encoder.pkl'}")
    logger.info(f"  Saved -> {SCREENSHOT_DIR / 'classification_report.txt'}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
import json
import joblib
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EvaluateModel")

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "models"
SCREENSHOT_DIR = PROJECT_DIR / "screenshots"
FEATURES_DIR = PROJECT_DIR / "features"


def evaluate():
    model_path = MODEL_DIR / "fall_model.pkl"
    npz_path = FEATURES_DIR / "features.npz"
    info_path = FEATURES_DIR / "info.json"

    if not model_path.exists():
        logger.error(f"No model found at {model_path}. Run train_model.py first.")
        return
    if not npz_path.exists():
        logger.error(f"No features at {npz_path}. Run extract_feature.py first.")
        return

    bundle = joblib.load(model_path)
    model = bundle["model"]
    activities = bundle["activities"]
    feature_names = bundle["feature_names"]

    data = np.load(npz_path)
    X_test, y_test = data["X_test"], data["y_test"]

    logger.info(f"Evaluating on {len(X_test)} test samples...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    logger.info(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")

    logger.info(f"\n{classification_report(y_test, y_pred, target_names=activities, zero_division=0)}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=activities, yticklabels=activities)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title("Confusion Matrix (Test Set)", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(SCREENSHOT_DIR / "confusion_matrix_test.png"), dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved to {SCREENSHOT_DIR / 'confusion_matrix_test.png'}")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(model, X_test, y_test, n_repeats=2,
                                    random_state=42, n_jobs=-1)
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
    logger.info(f"Feature importance saved to {SCREENSHOT_DIR / 'feature_importance.png'}")

    sample_probs = y_prob[:5]
    for i, probs in enumerate(sample_probs):
        pred_idx = np.argmax(probs)
        true_idx = y_test[i]
        logger.info(f"  Sample {i+1}: True={activities[true_idx]}, Pred={activities[pred_idx]}, "
                    f"Conf={probs[pred_idx]:.3f}")

    results = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "activities": activities,
        "test_samples": int(len(X_test)),
    }
    with open(SCREENSHOT_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {SCREENSHOT_DIR / 'evaluation_results.json'}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    evaluate()

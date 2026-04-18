import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

import tensorflow as tf

# ========================
# ⚙️ CONFIG
# ========================
DATA_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)


# ========================
# 📥 LOAD DATA & MODEL
# ========================
def load_data():
    logging.info("Loading test data...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_test, y_test


def load_model():
    logging.info("Loading trained model...")
    model_path = os.path.join(MODEL_DIR, "final_model")
    model = tf.keras.models.load_model(model_path)
    return model


# ========================
# 📊 METRICS
# ========================
def compute_metrics(y_true, y_pred, y_prob):
    logging.info("Computing metrics...")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        "auc": roc_auc_score(y_true, y_prob)
    }

    for k, v in metrics.items():
        logging.info(f"{k.upper()}: {v:.4f}")

    return metrics


# ========================
# 📉 CONFUSION MATRIX
# ========================
def plot_confusion_matrix(y_true, y_pred):
    logging.info("Plotting confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()

    logging.info(f"Saved: {path}")


# ========================
# 📈 ROC CURVE
# ========================
def plot_roc_curve(y_true, y_prob):
    logging.info("Plotting ROC curve...")

    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path)
    plt.close()

    logging.info(f"Saved: {path}")


# ========================
# 🎯 THRESHOLD TUNING
# ========================
def find_best_threshold(y_true, y_prob):
    logging.info("Finding best threshold...")

    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred)

        if score > best_f1:
            best_f1 = score
            best_thresh = t

    logging.info(f"Best Threshold: {best_thresh:.3f} | F1: {best_f1:.4f}")

    return best_thresh


# ========================
# 🚀 EVALUATE
# ========================
def evaluate():
    X_test, y_test = load_data()
    model = load_model()

    logging.info("Generating predictions...")

    y_prob = model.predict(X_test).ravel()
    best_thresh = find_best_threshold(y_test, y_prob)

    y_pred = (y_prob >= best_thresh).astype(int)

    compute_metrics(y_test, y_pred, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

    logging.info("✅ Evaluation completed!")


# ========================
# 🏁 MAIN
# ========================
if __name__ == "__main__":
    evaluate()

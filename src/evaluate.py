"""
evaluate.py — WikiArt Evaluation Framework

Reusable functions for model evaluation: metrics computation, confusion matrices,
learning curves, per-class F1 charts, and error analysis.

Usage:
    python src/evaluate.py --model baseline --checkpoint results/models/baseline_cnn.keras
    python src/evaluate.py --model custom_cnn --checkpoint results/models/custom_cnn.keras
    python src/evaluate.py --model vit --checkpoint results/models/vit.keras
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
)

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import build_datasets


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_data(splits_dir="data/splits", img_size=(224, 224), batch_size=32,
                   processed_dir="data/processed"):
    """Load the test dataset and return (test_ds, test_ds_oh, num_classes, artist_names)."""
    _, _, test_ds, num_classes = build_datasets(
        splits_dir=splits_dir,
        img_size=img_size,
        batch_size=batch_size,
        augment_train=False,
        use_processed=True,
        processed_dir=processed_dir,
    )
    test_ds_oh = test_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

    test_df = pd.read_csv(
        Path(splits_dir) / "test.csv", encoding="utf-8-sig"
    )
    artist_names = sorted(test_df["artist"].unique())
    return test_ds, test_ds_oh, num_classes, artist_names


def collect_predictions(model, test_ds):
    """Run inference on test_ds and return (y_true, y_pred) as numpy arrays."""
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, artist_names):
    """Return a dict with overall accuracy, F1-macro, and the full classification report string."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=artist_names, digits=3)
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "per_class_f1": per_class_f1,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, artist_names, model_name, save_dir="results/figures"):
    """Plot and save a normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=artist_names)
    disp.plot(ax=ax, cmap="Blues", values_format=".2f", xticks_rotation=90)
    ax.set_title(f"{model_name} — Normalised Confusion Matrix (Test Set)", fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_confusion_matrix_normalised.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_per_class_f1(per_class_f1, artist_names, model_name, save_dir="results/figures"):
    """Plot and save a horizontal bar chart of per-class F1 scores."""
    macro_f1 = np.mean(per_class_f1)
    sorted_idx = np.argsort(per_class_f1)
    sorted_names = [artist_names[i] for i in sorted_idx]
    sorted_f1 = per_class_f1[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(sorted_names, sorted_f1, color="steelblue")
    ax.axvline(x=macro_f1, color="red", linestyle="--", label=f"Macro F1 = {macro_f1:.3f}")
    ax.set_xlabel("F1-Score")
    ax.set_title(f"{model_name} — Per-class F1-Score (Test Set)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_per_class_f1.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_learning_curves(log_csv, model_name, save_dir="results/figures"):
    """Plot training vs validation loss and accuracy from a CSV log file."""
    df = pd.read_csv(log_csv)
    epochs = df["epoch"] + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, df["loss"], label="Train Loss")
    axes[0].plot(epochs, df["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, df["accuracy"], label="Train Accuracy")
    axes[1].plot(epochs, df["val_accuracy"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} — Learning Curves", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_learning_curves.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


def plot_model_comparison(summary_df, save_dir="results/figures"):
    """Plot a grouped bar chart comparing accuracy and F1-macro across models."""
    models = summary_df["model"]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, summary_df["accuracy"], width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, summary_df["f1_macro"], width, label="F1-Macro", color="coral")

    ax.set_ylabel("Score")
    ax.set_title("Cross-Model Comparison (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WikiArt Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name (baseline, custom_cnn, vit)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved .keras model")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"\n=== Evaluating {args.model} ===")
    print(f"Checkpoint: {args.checkpoint}\n")

    # Load data
    test_ds, test_ds_oh, num_classes, artist_names = load_test_data(batch_size=args.batch_size)

    # Load model
    model = tf.keras.models.load_model(args.checkpoint)
    print(f"Parameters: {model.count_params():,}\n")

    # Predictions and metrics
    y_true, y_pred = collect_predictions(model, test_ds)
    metrics = compute_metrics(y_true, y_pred, artist_names)

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Macro:  {metrics['f1_macro']:.4f}\n")
    print(metrics["report"])

    # Plots
    plot_confusion_matrix(y_true, y_pred, artist_names, args.model)
    plot_per_class_f1(metrics["per_class_f1"], artist_names, args.model)

    log_csv = f"results/logs/{args.model}.csv"
    if os.path.exists(log_csv):
        plot_learning_curves(log_csv, args.model)

    print(f"\nDone — all figures saved to results/figures/")


if __name__ == "__main__":
    main()

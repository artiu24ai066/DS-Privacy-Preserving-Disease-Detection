"""
evaluate.py
-----------
Shared evaluation utilities used by all experiments.
Computes accuracy, precision, recall, F1, specificity, ROC-AUC,
confusion matrix, and generates standard plots.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Core evaluation ───────────────────────────────────────────────────────────
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Runs inference on all batches in loader and returns a metrics dict.
    Metrics: accuracy, precision, recall, f1, specificity, roc_auc.
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, 1]   # P(PNEUMONIA)
            preds   = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Confusion matrix — handle the case where the model predicts only one class
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm is always 2×2 when labels=[0,1] is specified explicitly
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC-AUC requires both classes present in y_true
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "roc_auc":     roc_auc,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }
    return metrics


# ── Confusion matrix plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(metrics: dict, title: str, save_name: str):
    """Saves a confusion matrix heatmap to results/plots/."""
    cm = np.array([[metrics["TN"], metrics["FP"]],
                   [metrics["FN"], metrics["TP"]]])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NORMAL", "PNEUMONIA"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {path}")


# ── Metrics bar chart ─────────────────────────────────────────────────────────
def plot_experiment_comparison(results_csv: str, save_name: str = "experiment_comparison.png"):
    """
    Reads results/metrics.csv and generates a grouped bar chart
    comparing accuracy, recall, and F1 across experiments.
    """
    df = pd.read_csv(results_csv)

    metrics_to_plot = ["accuracy", "recall", "f1"]
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, df[metric], width, label=metric.capitalize())

    ax.set_xticks(x + width)
    ax.set_xticklabels(df["experiment"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison: Accuracy, Recall, F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison chart → {path}")


# ── Training-time bar chart ───────────────────────────────────────────────────
def plot_training_times(results_csv: str, save_name: str = "training_times.png"):
    df = pd.read_csv(results_csv)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["experiment"], df["training_time_s"], color="steelblue")
    ax.set_xlabel("Training time (seconds)")
    ax.set_title("Training Time Comparison")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved training time chart → {path}")


# ── Results CSV helper ────────────────────────────────────────────────────────
def save_result(row: dict, csv_path: str):
    """
    Writes one experiment result to the master CSV.
    If the file already exists and contains a row with the same
    'experiment' value, the old row is replaced rather than duplicated.
    Creates the file with headers if it does not exist.
    """
    df_new = pd.DataFrame([row])
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # Drop any pre-existing rows for this experiment
        if "experiment" in df_existing.columns and "experiment" in row:
            df_existing = df_existing[
                df_existing["experiment"] != row["experiment"]
            ]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"  Result saved → {csv_path}")


# ── Pretty print metrics ──────────────────────────────────────────────────────
def print_metrics(metrics: dict, label: str = ""):
    print(f"\n  {'─'*40}")
    if label:
        print(f"  {label}")
    print(f"  {'─'*40}")
    for k, v in metrics.items():
        if k not in ("TP", "TN", "FP", "FN"):
            print(f"  {k:<15} {v:.4f}")
    print(f"  TP={metrics['TP']}  TN={metrics['TN']}  "
          f"FP={metrics['FP']}  FN={metrics['FN']}")
    print(f"  {'─'*40}")

"""
visualize.py
------------
Generates all final visualizations from results/metrics.csv.

Run this AFTER all experiments have been completed.
Saves all plots to results/plots/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
import config
from config import TRAIN_DIR, TEST_DIR
from evaluate import PLOTS_DIR

RESULTS_CSV = config.METRICS_CSV


def load_results() -> pd.DataFrame:
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_CSV}\n"
            "Run all experiments first using run_all.py"
        )
    return pd.read_csv(RESULTS_CSV)


# ── Plot 1: Experiment comparison bar chart ───────────────────────────────────
def plot_comparison(df: pd.DataFrame):
    metrics = ["accuracy", "precision", "recall", "f1"]
    x       = np.arange(len(df))
    width   = 0.20

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df[m], width, label=m.capitalize(), color=c, alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(df["experiment"], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score")
    ax.set_title("Experiment Comparison: Accuracy, Precision, Recall, F1")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "1_experiment_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Plot 2: Training time comparison ─────────────────────────────────────────
def plot_training_time(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = plt.cm.Blues(np.linspace(0.4, 0.9, len(df)))
    bars    = ax.barh(df["experiment"], df["training_time_s"], color=colors)

    for bar, val in zip(bars, df["training_time_s"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.0f}s", va="center", fontsize=9)

    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time Across Experiments")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "2_training_times.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Plot 3: Reservoir sampling — performance vs data size ─────────────────────
def plot_reservoir_effect(df: pd.DataFrame):
    """Shows how F1 and recall change with sampling fraction."""
    reservoir_experiments = [
        "E3 - Federated + DP",
        "E4a - FL+DP+RS50%",
        "E4b - FL+DP+RS25%",
    ]
    rs_rows = df[df["experiment"].isin(reservoir_experiments)].copy()
    rs_rows["sampling_frac"] = pd.to_numeric(rs_rows["sampling_frac"])
    rs_rows = rs_rows.sort_values("sampling_frac", ascending=False)

    if len(rs_rows) < 2:
        print("  Not enough reservoir experiments to plot (need E3, E4a, E4b).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric, color in zip(axes, ["f1", "recall"], ["steelblue", "crimson"]):
        ax.plot(rs_rows["sampling_frac"], rs_rows[metric],
                marker="o", linewidth=2, color=color, markersize=8)
        for _, row in rs_rows.iterrows():
            ax.annotate(
                f"{row[metric]:.3f}",
                (row["sampling_frac"], row[metric]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8,
            )
        ax.set_xlabel("Sampling Fraction")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs Reservoir Size")
        ax.set_xlim(-0.05, 1.10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

    plt.suptitle("Effect of Reservoir Sampling on Model Performance (FL + DP)")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "3_reservoir_effect.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Plot 4: Privacy budget vs F1 ─────────────────────────────────────────────
def plot_privacy_vs_performance(df: pd.DataFrame):
    dp_df = df[df["dp_enabled"] == True].dropna(subset=["epsilon"])

    if len(dp_df) < 2:
        print("  Not enough DP experiments to plot privacy vs performance.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(dp_df["epsilon"], dp_df["f1"], s=100, color="darkorange", zorder=5)

    for _, row in dp_df.iterrows():
        ax.annotate(
            row["experiment"],
            (row["epsilon"], row["f1"]),
            textcoords="offset points", xytext=(5, 5), fontsize=8,
        )

    ax.set_xlabel("Privacy Budget ε (lower = stronger privacy)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Privacy Budget vs Diagnostic Performance")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "4_privacy_vs_f1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Plot 5: Recall comparison (most important clinical metric) ────────────────
def plot_recall_focus(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = ["#2ecc71" if r >= 0.85 else "#e74c3c" for r in df["recall"]]
    bars    = ax.bar(df["experiment"], df["recall"], color=colors, alpha=0.85)

    ax.axhline(0.85, color="gray", linestyle="--", linewidth=1,
               label="Reference line (0.85)")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Recall (Sensitivity)")
    ax.set_title("Recall Across Experiments\n"
                 "(Recall = fraction of pneumonia cases correctly detected)")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["experiment"], rotation=20, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, df["recall"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "5_recall_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Plot 6: Class distribution (EDA helper) ───────────────────────────────────
def plot_class_distribution(train_dir: str = None, test_dir: str = None):
    """Called separately from the EDA notebook, but also usable standalone."""
    if train_dir is None:
        train_dir = TRAIN_DIR
    if test_dir is None:
        test_dir = TEST_DIR

    from data_loader import ChestXRayDataset, make_train_val_split

    raw_train = ChestXRayDataset(train_dir, transform=None)
    labels    = raw_train.get_labels()
    train_idx, val_idx = make_train_val_split(train_dir, 0.20)

    test_ds = ChestXRayDataset(test_dir, transform=None)
    test_labels = test_ds.get_labels()

    splits = {
        "Train":      [labels[i] for i in train_idx],
        "Validation": [labels[i] for i in val_idx],
        "Test":       test_labels,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (split_name, split_labels) in zip(axes, splits.items()):
        counts  = [split_labels.count(0), split_labels.count(1)]
        classes = ["NORMAL", "PNEUMONIA"]
        bars    = ax.bar(classes, counts, color=["#3498db", "#e74c3c"], alpha=0.85)
        ax.set_title(f"{split_name} Set")
        ax.set_ylabel("Number of Images")
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha="center", fontsize=11)

    plt.suptitle("Class Distribution Across Splits", fontsize=13)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "0_class_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary_table(df: pd.DataFrame):
    cols = ["experiment", "n_train_images", "accuracy", "precision",
            "recall", "f1", "roc_auc", "epsilon", "training_time_s"]
    present = [c for c in cols if c in df.columns]
    print("\n" + "="*90)
    print("RESULTS SUMMARY")
    print("="*90)
    print(df[present].to_string(index=False))
    print("="*90 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_class_distribution()   # uses config paths by default
    df = load_results()
    print_summary_table(df)
    plot_comparison(df)
    plot_training_time(df)
    plot_reservoir_effect(df)
    plot_privacy_vs_performance(df)
    plot_recall_focus(df)
    print("\nAll plots saved to results/plots/")

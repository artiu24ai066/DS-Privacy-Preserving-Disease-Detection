"""
eda.py  — Exploratory Data Analysis
------------------------------------
Stage 1 of the DS project pipeline.

Run this script (or copy cells into a Jupyter notebook) to:
  1. Summarize the dataset (image counts per class/split)
  2. Plot the class distribution
  3. Display sample images from each class
  4. Show pixel intensity statistics

DS Unit 1 (Lifecycle) — This is the "Data Understanding" phase.
DS Unit 2 (Statistics) — Class distribution, mean/std of pixel values.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import (
    ChestXRayDataset, dataset_summary,
    make_train_val_split, get_eval_transform,
)

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
TRAIN_DIR = os.path.join(BASE_DIR, "chest_xray", "train")
TEST_DIR  = os.path.join(BASE_DIR, "chest_xray", "chest_xray", "test")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── 1. Dataset Summary ─────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  STAGE 1 — Exploratory Data Analysis")
print("="*50)

df_summary = dataset_summary(TRAIN_DIR, TEST_DIR, val_fraction=0.20)


# ── 2. Class Distribution Plot ────────────────────────────────────────────────
train_ds  = ChestXRayDataset(TRAIN_DIR, transform=None)
labels    = train_ds.get_labels()
train_idx, val_idx = make_train_val_split(TRAIN_DIR, 0.20)
test_ds   = ChestXRayDataset(TEST_DIR,  transform=None)
test_lbls = test_ds.get_labels()

splits = {
    "Train":      [labels[i] for i in train_idx],
    "Validation": [labels[i] for i in val_idx],
    "Test":       test_lbls,
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (split_name, split_labels) in zip(axes, splits.items()):
    counts  = [split_labels.count(0), split_labels.count(1)]
    total   = sum(counts)
    classes = ["NORMAL", "PNEUMONIA"]
    bars    = ax.bar(classes, counts, color=["#3498db", "#e74c3c"], alpha=0.85)
    ax.set_title(f"{split_name} (n={total})")
    ax.set_ylabel("Images")
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 15,
                f"{count}\n({pct:.1f}%)", ha="center", fontsize=9)

plt.suptitle("Class Distribution Across Dataset Splits", fontsize=13)
plt.tight_layout()
path = os.path.join(PLOTS_DIR, "0_class_distribution.png")
fig.savefig(path, dpi=150)
plt.show()
print(f"\nSaved → {path}")

# ── 3. Sample Images ─────────────────────────────────────────────────────────
n_samples = 4
fig, axes = plt.subplots(2, n_samples, figsize=(13, 6))
fig.suptitle("Sample Chest X-ray Images", fontsize=13)

for class_idx, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
    class_samples = [(p, l) for p, l in train_ds.samples if l == class_idx][:n_samples]
    for col, (img_path, _) in enumerate(class_samples):
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        axes[class_idx][col].imshow(img, cmap="gray")
        axes[class_idx][col].set_title(class_name, fontsize=10)
        axes[class_idx][col].axis("off")

plt.tight_layout()
path = os.path.join(PLOTS_DIR, "0_sample_images.png")
fig.savefig(path, dpi=150)
plt.show()
print(f"Saved → {path}")


# ── 4. Pixel Statistics ───────────────────────────────────────────────────────
print("\n  Computing pixel statistics on a sample of training images...")
transform = get_eval_transform()

means, stds = [], []
sample_size = min(200, len(train_ds))
sample_indices = train_idx[:sample_size]

for idx in sample_indices:
    img_path, _ = train_ds.samples[idx]
    img   = Image.open(img_path).convert("RGB")
    img_t = transform(img)           # (3, 224, 224) tensor
    means.append(img_t.mean(dim=[1, 2]).numpy())
    stds.append(img_t.std(dim=[1, 2]).numpy())

means = np.array(means)    # (N, 3)
stds  = np.array(stds)

channels = ["R", "G", "B"]
print(f"\n  Pixel statistics (after normalization, sample={sample_size}):")
print(f"  {'Channel':<10} {'Mean':>8} {'Std':>8}")
print(f"  {'─'*28}")
for c, ch in enumerate(channels):
    print(f"  {ch:<10} {means[:, c].mean():>8.4f} {stds[:, c].mean():>8.4f}")

# Image size distribution
print("\n  Checking raw image sizes (first 50 training images)...")
sizes = []
for idx in train_idx[:50]:
    img_path, _ = train_ds.samples[idx]
    w, h = Image.open(img_path).size
    sizes.append((w, h))

widths  = [s[0] for s in sizes]
heights = [s[1] for s in sizes]
print(f"  Width  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
print(f"  Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")
print("\n  All images will be resized to 224×224 before training.")
print("\n  EDA complete. Proceed to train_centralized.py (E1).")

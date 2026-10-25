"""
train_centralized.py
--------------------
Experiment E1 — Centralized baseline.

Trains MobileNetV2 on the full training set with no federated learning
and no differential privacy. Records accuracy, precision, recall, F1,
specificity, ROC-AUC, confusion matrix, and training time.

This result serves as the upper-bound reference for all other experiments.
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import get_loaders, dataset_summary
from model       import build_model, count_parameters
from evaluate    import (
    evaluate_model, print_metrics,
    plot_confusion_matrix, save_result,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
TRAIN_DIR = os.path.join(BASE_DIR, "chest_xray", "train")
TEST_DIR  = os.path.join(BASE_DIR, "chest_xray", "chest_xray", "test")
CSV_PATH  = os.path.join(BASE_DIR, "results", "metrics.csv")

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
EPOCHS        = 5
VAL_FRACTION  = 0.20
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Standard training loop for one epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def run():
    print("\n" + "="*55)
    print("  EXPERIMENT E1 — Centralized Baseline")
    print("="*55)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")

    # Dataset
    dataset_summary(TRAIN_DIR, TEST_DIR, VAL_FRACTION)
    train_loader, val_loader, test_loader, _ = get_loaders(
        TRAIN_DIR, TEST_DIR,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    # Model
    model = build_model(num_classes=2, freeze_backbone=True).to(DEVICE)
    print(f"\n  Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # Training loop
    best_val_f1  = 0.0
    best_weights = None
    start_time   = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_metrics = evaluate_model(model, val_loader, DEVICE)

        print(f"  Epoch {epoch}/{EPOCHS}  "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  "
              f"val_recall={val_metrics['recall']:.4f}  "
              f"val_f1={val_metrics['f1']:.4f}")

        # Keep best checkpoint by validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            import copy
            best_weights = copy.deepcopy(model.state_dict())

    training_time = time.time() - start_time

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(best_weights)
    test_metrics = evaluate_model(model, test_loader, DEVICE)
    print_metrics(test_metrics, "E1 — Test Set Results")

    # Save confusion matrix
    plot_confusion_matrix(
        test_metrics,
        title="E1 — Centralized Baseline",
        save_name="cm_e1_centralized.png",
    )

    # Save metrics
    row = {
        "experiment":      "E1 - Centralized",
        "n_clients":       1,
        "fl_rounds":       0,
        "dp_enabled":      False,
        "noise_multiplier": None,
        "epsilon":         None,
        "sampling_frac":   1.0,
        "n_train_images":  len(train_loader.dataset),
        "training_time_s": round(training_time, 1),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in test_metrics.items()},
    }
    save_result(row, CSV_PATH)

    print(f"\n  Total training time : {training_time:.1f}s")
    print("  E1 complete.\n")
    return test_metrics


if __name__ == "__main__":
    run()

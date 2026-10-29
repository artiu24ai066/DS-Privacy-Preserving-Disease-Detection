"""
train_centralized.py
--------------------
Experiment E1 — Centralized Baseline.

Trains MobileNetV2 on the full training partition with no federated
learning and no differential privacy.  Records all metrics and saves
the best checkpoint (by validation F1).

Architecture note
-----------------
E1 uses freeze_backbone=True (standard transfer learning).
E2/E3/E4 use the same architecture built with freeze_backbone=False
then manually frozen, so state-dict keys are identical across all
experiments.  Both configurations freeze the MobileNetV2 feature
extractor and train only the classifier head; the key difference
is that E1's backbone parameters are absent from the optimizer
while the federated experiments' backbone parameters are present
but have requires_grad=False.  This means FedAvg correctly averages
the full state-dict for the federated experiments.
"""

import copy
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

import config
from config import (
    TRAIN_DIR, TEST_DIR, METRICS_CSV, CHECKPOINT_DIR,
    BATCH_SIZE, LEARNING_RATE, SEED, VAL_FRACTION,
    UNFREEZE_LAST_N, make_output_dirs,
)
from data_loader import get_loaders, dataset_summary, get_class_weights
from model       import build_model, count_parameters
from evaluate    import evaluate_model, print_metrics, plot_confusion_matrix, save_result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One epoch of standard SGD.  Backbone kept in eval() to avoid
    updating frozen BatchNorm running statistics."""
    model.train()
    model.features.eval()
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
    make_output_dirs()

    print("\n" + "=" * 55)
    print("  EXPERIMENT E1 — Centralized Baseline")
    print("=" * 55)
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")
    print(f"  Batch  : {BATCH_SIZE}")
    print(f"  LR     : {LEARNING_RATE}")
    print(f"  Seed   : {SEED}")

    dataset_summary(TRAIN_DIR, TEST_DIR, VAL_FRACTION)

    train_loader, val_loader, test_loader, _ = get_loaders(
        TRAIN_DIR, TEST_DIR,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    model = build_model(num_classes=2, freeze_backbone=True,
                        unfreeze_last_n=UNFREEZE_LAST_N).to(DEVICE)
    print(f"\n  Trainable parameters: {count_parameters(model):,}"
          f"  (unfreeze_last_n={UNFREEZE_LAST_N})")

    # Class-weighted loss to counteract the 74.3% PNEUMONIA imbalance.
    # Without weights, CE loss is minimised by predicting PNEUMONIA for all
    # images (the majority class), collapsing specificity to 0.
    from data_loader import ChestXRayDataset, make_train_val_split
    raw_ds    = ChestXRayDataset(TRAIN_DIR, transform=None)
    all_lbl   = raw_ds.get_labels()
    tr_idx, _ = make_train_val_split(TRAIN_DIR, VAL_FRACTION, SEED)
    class_weights = get_class_weights(TRAIN_DIR, tr_idx, all_lbl, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
    )

    best_val_f1  = float("-inf")
    best_weights = None
    checkpoint_path = str(CHECKPOINT_DIR / "e1_centralized_best.pt")
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

        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, checkpoint_path)
            print(f"  Saved best checkpoint (val F1={best_val_f1:.4f}) → {checkpoint_path}")

    training_time = time.time() - start_time

    if best_weights is None:
        raise RuntimeError("No checkpoint saved — check training loop.")
    model.load_state_dict(best_weights)

    # Evaluate once on the test set using the best validation checkpoint
    test_metrics = evaluate_model(model, test_loader, DEVICE)
    print_metrics(test_metrics, "E1 — Test Set Results")
    plot_confusion_matrix(
        test_metrics,
        title="E1 — Centralized Baseline",
        save_name="cm_e1_centralized.png",
    )

    row = {
        "experiment":         "E1 - Centralized",
        "n_clients":          1,
        "fl_rounds":          0,
        "local_epochs":       EPOCHS,
        "batch_size":         BATCH_SIZE,
        "learning_rate":      LEARNING_RATE,
        "dp_enabled":         False,
        "noise_multiplier":   None,
        "max_grad_norm":      None,
        "target_delta":       None,
        "epsilon":            None,
        "client_epsilons":    None,
        "privacy_assumption": None,
        "sampling_frac":      1.0,
        "n_train_images":     len(train_loader.dataset),
        "training_time_s":    round(training_time, 1),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in test_metrics.items()},
    }
    save_result(row, METRICS_CSV)

    print(f"\n  Total training time : {training_time:.1f}s")
    print("  E1 complete.\n")
    return test_metrics


if __name__ == "__main__":
    config.verify_paths()
    run()

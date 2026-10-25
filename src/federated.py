"""
federated.py
------------
Experiments E2, E3, E4 — Federated Learning simulation.

E2 : Federated learning, no differential privacy
E3 : Federated learning + DP-SGD (Opacus)
E4 : Federated learning + DP-SGD + Reservoir Sampling (Unit 5 extension)

Architecture:
  - 3 simulated clients (hospitals), each with a local partition of training data
  - Custom FedAvg aggregation (weighted by number of local samples)
  - Opacus PrivacyEngine wraps local optimizer for DP experiments
  - Reservoir sampling reduces local data before training (E4)

DS connections:
  - Unit 3 (Distributed Processing): each client = distributed node
  - Unit 5 (Data Stream Mining):      reservoir sampling on client data
  - Unit 2 (Statistics & Evaluation): privacy budget ε, all classification metrics
"""

import os
import sys
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (
    ChestXRayDataset, get_train_transform, get_eval_transform,
    make_train_val_split, partition_for_clients,
)
from model    import build_model, count_parameters
from evaluate import (
    evaluate_model, print_metrics,
    plot_confusion_matrix, save_result,
)
from reservoir import sample_client_data

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
TRAIN_DIR = os.path.join(BASE_DIR, "chest_xray", "train")
TEST_DIR  = os.path.join(BASE_DIR, "chest_xray", "chest_xray", "test")
CSV_PATH  = os.path.join(BASE_DIR, "results", "metrics.csv")

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_CLIENTS     = 3
FL_ROUNDS     = 3      # communication rounds (increase if you have more time)
LOCAL_EPOCHS  = 2      # epochs each client trains per round
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
VAL_FRACTION  = 0.20
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DP hyperparameters (used in E3 / E4)
NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM    = 1.0
TARGET_DELTA     = 1e-5


# ── FedAvg ────────────────────────────────────────────────────────────────────
def fedavg(global_model: nn.Module, client_updates: list[tuple[dict, int]]) -> nn.Module:
    """
    Federated Averaging (McMahan et al. 2017).

    Aggregates client model weights weighted by their local dataset size.

    Args:
        global_model   : the current server model (used for structure only)
        client_updates : list of (state_dict, n_samples) from each client

    Returns:
        Updated global model.
    """
    total_samples = sum(n for _, n in client_updates)
    avg_state     = copy.deepcopy(client_updates[0][0])

    # Initialise accumulator to zero
    for key in avg_state:
        avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)

    # Weighted sum
    for state_dict, n_samples in client_updates:
        weight = n_samples / total_samples
        for key in avg_state:
            avg_state[key] += state_dict[key].float() * weight

    new_model = copy.deepcopy(global_model)
    new_model.load_state_dict(avg_state)
    return new_model


# ── Single-client local training (no DP) ─────────────────────────────────────
def local_train(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epochs:    int = LOCAL_EPOCHS,
) -> tuple[nn.Module, int]:
    """
    Standard local training (used in E2).
    Returns the trained model and number of local samples.
    """
    local_model = copy.deepcopy(model).to(device)
    optimizer   = Adam(
        filter(lambda p: p.requires_grad, local_model.parameters()),
        lr=LEARNING_RATE,
    )
    local_model.train()

    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(images), labels)
            loss.backward()
            optimizer.step()

    return local_model, len(loader.dataset)


# ── Single-client local training WITH DP ─────────────────────────────────────
def local_train_dp(
    model:            nn.Module,
    loader:           DataLoader,
    criterion:        nn.Module,
    device:           torch.device,
    epochs:           int   = LOCAL_EPOCHS,
    noise_multiplier: float = NOISE_MULTIPLIER,
    max_grad_norm:    float = MAX_GRAD_NORM,
    target_delta:     float = TARGET_DELTA,
) -> tuple[nn.Module, int, float]:
    """
    DP-SGD local training using Opacus (used in E3 / E4).

    Returns:
        (trained_model, n_samples, epsilon)
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError:
        raise ImportError(
            "Opacus is required for DP training. "
            "Install it with: pip install opacus"
        )

    local_model = copy.deepcopy(model).to(device)

    # Opacus requires BatchNorm → GroupNorm replacement
    if not ModuleValidator.is_valid(local_model):
        local_model = ModuleValidator.fix(local_model)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, local_model.parameters()),
        lr=LEARNING_RATE,
    )

    privacy_engine = PrivacyEngine()
    local_model, optimizer, dp_loader = privacy_engine.make_private(
        module=local_model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=False,
    )

    local_model.train()
    for _ in range(epochs):
        for images, labels in dp_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(images), labels)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)

    # Unwrap the Opacus wrapper to get a plain nn.Module
    plain_model = local_model._module if hasattr(local_model, "_module") else local_model

    return plain_model, len(loader.dataset), epsilon


# ── Federated training orchestrator ──────────────────────────────────────────
def run_federated(
    experiment_id:    str,
    use_dp:           bool  = False,
    sampling_frac:    float = 1.0,
    noise_multiplier: float = NOISE_MULTIPLIER,
):
    """
    Runs one federated experiment and saves results.

    Args:
        experiment_id    : e.g. "E2 - Federated"
        use_dp           : whether to apply DP-SGD locally
        sampling_frac    : reservoir sampling fraction per client (1.0 = no sampling)
        noise_multiplier : DP noise parameter (ignored if use_dp=False)
    """
    print("\n" + "="*55)
    print(f"  EXPERIMENT {experiment_id}")
    if use_dp:
        print(f"  DP-SGD  noise={noise_multiplier}  clip={MAX_GRAD_NORM}  δ={TARGET_DELTA}")
    if sampling_frac < 1.0:
        print(f"  Reservoir sampling fraction = {sampling_frac:.0%}")
    print("="*55)
    print(f"  Device   : {DEVICE}")
    print(f"  Clients  : {N_CLIENTS}")
    print(f"  Rounds   : {FL_ROUNDS}")
    print(f"  Local EP : {LOCAL_EPOCHS}")

    # ── Load datasets ──────────────────────────────────────────────────────────
    # Two separate dataset objects: one with train transform, one without
    # (we need the non-transform version to get labels for partitioning)
    raw_train_ds = ChestXRayDataset(TRAIN_DIR, transform=None)
    all_labels   = raw_train_ds.get_labels()

    train_idx, val_idx = make_train_val_split(TRAIN_DIR, VAL_FRACTION, SEED)

    # Partition training indices among clients
    client_partitions = partition_for_clients(
        train_idx, all_labels, n_clients=N_CLIENTS, seed=SEED
    )

    # Apply reservoir sampling if requested
    if sampling_frac < 1.0:
        print(f"\n  Applying reservoir sampling ({sampling_frac:.0%}) to each client:")
        client_partitions = [
            sample_client_data(
                indices=part,
                labels=all_labels,
                fraction=sampling_frac,
                seed=SEED + cid,          # different seed per client
            )
            for cid, part in enumerate(client_partitions)
        ]

    # Print per-client sizes
    for cid, part in enumerate(client_partitions):
        n_norm = sum(1 for i in part if all_labels[i] == 0)
        n_pneu = sum(1 for i in part if all_labels[i] == 1)
        print(f"  Client {cid+1}: {len(part)} images  "
              f"(NORMAL={n_norm}, PNEUMONIA={n_pneu})")

    # Build DataLoaders for each client
    train_ds_aug = ChestXRayDataset(TRAIN_DIR, transform=get_train_transform())

    def make_client_loader(indices):
        subset = Subset(train_ds_aug, indices)
        return DataLoader(subset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, drop_last=True)

    client_loaders = [make_client_loader(p) for p in client_partitions]

    # Validation + test loaders (shared, same for all experiments)
    val_ds = ChestXRayDataset(TRAIN_DIR, transform=get_eval_transform())
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_ds = ChestXRayDataset(TEST_DIR, transform=get_eval_transform())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    # ── Federated loop ─────────────────────────────────────────────────────────
    global_model  = build_model(num_classes=2, freeze_backbone=True).to(DEVICE)
    criterion     = nn.CrossEntropyLoss()
    epsilons      = []      # collected ε per round (DP only)

    start_time = time.time()

    for rnd in range(1, FL_ROUNDS + 1):
        print(f"\n  ── Round {rnd}/{FL_ROUNDS} ──")
        client_updates = []

        for cid, loader in enumerate(client_loaders):
            if len(loader.dataset) == 0:
                print(f"    Client {cid+1}: skipped (empty after sampling)")
                continue

            if use_dp:
                local_model, n_samples, eps = local_train_dp(
                    global_model, loader, criterion, DEVICE,
                    noise_multiplier=noise_multiplier,
                )
                epsilons.append(eps)
                print(f"    Client {cid+1}: {n_samples} samples  ε={eps:.4f}")
            else:
                local_model, n_samples = local_train(
                    global_model, loader, criterion, DEVICE
                )
                print(f"    Client {cid+1}: {n_samples} samples")

            client_updates.append((local_model.state_dict(), n_samples))

        # FedAvg aggregation
        global_model = fedavg(global_model, client_updates)
        global_model = global_model.to(DEVICE)

        # Validation check after each round
        val_metrics = evaluate_model(global_model, val_loader, DEVICE)
        print(f"  Round {rnd} val — acc={val_metrics['accuracy']:.4f}  "
              f"recall={val_metrics['recall']:.4f}  f1={val_metrics['f1']:.4f}")

    training_time = time.time() - start_time

    # ── Final test evaluation ──────────────────────────────────────────────────
    test_metrics = evaluate_model(global_model, test_loader, DEVICE)
    print_metrics(test_metrics, f"{experiment_id} — Test Set Results")

    # Confusion matrix
    safe_name = experiment_id.lower().replace(" ", "_").replace("-", "").replace("/", "")
    plot_confusion_matrix(
        test_metrics,
        title=experiment_id,
        save_name=f"cm_{safe_name}.png",
    )

    # Compute reported ε (average over rounds and clients, DP only)
    reported_eps = round(float(sum(epsilons) / len(epsilons)), 4) if epsilons else None

    total_sampled = sum(len(p) for p in client_partitions)

    row = {
        "experiment":       experiment_id,
        "n_clients":        N_CLIENTS,
        "fl_rounds":        FL_ROUNDS,
        "dp_enabled":       use_dp,
        "noise_multiplier": noise_multiplier if use_dp else None,
        "epsilon":          reported_eps,
        "sampling_frac":    sampling_frac,
        "n_train_images":   total_sampled,
        "training_time_s":  round(training_time, 1),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in test_metrics.items()},
    }
    save_result(row, CSV_PATH)

    print(f"\n  Total training time : {training_time:.1f}s")
    print(f"  {experiment_id} complete.\n")
    return test_metrics, reported_eps


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # E2 — Federated, no DP
    run_federated("E2 - Federated No DP",
                  use_dp=False, sampling_frac=1.0)

    # E3 — Federated + DP-SGD
    run_federated("E3 - Federated + DP",
                  use_dp=True, sampling_frac=1.0,
                  noise_multiplier=NOISE_MULTIPLIER)

    # E4a — Federated + DP + Reservoir 50%
    run_federated("E4a - Federated + DP + RS50",
                  use_dp=True, sampling_frac=0.50,
                  noise_multiplier=NOISE_MULTIPLIER)

    # E4b — Federated + DP + Reservoir 25%
    run_federated("E4b - Federated + DP + RS25",
                  use_dp=True, sampling_frac=0.25,
                  noise_multiplier=NOISE_MULTIPLIER)

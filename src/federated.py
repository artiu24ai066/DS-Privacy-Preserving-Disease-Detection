"""
federated.py
------------
Experiments E2, E3, E4a, E4b — Federated Learning simulation.

E2  : Federated learning, no differential privacy
E3  : Federated learning + DP-SGD (Opacus)
E4a : Federated + DP-SGD + Reservoir Sampling at 50%
E4b : Federated + DP-SGD + Reservoir Sampling at 25%

Architecture
  - 3 simulated clients (hospitals), each with a disjoint local partition
  - Standard weighted FedAvg aggregation (McMahan et al. 2017)
  - Opacus PrivacyEngine for DP-SGD in E3/E4a/E4b
  - Reservoir sampling (Algorithm R) reduces client data in E4a/E4b

DP accounting note
  A single PrivacyEngine per client is created ONCE before the FL loop
  and reused across all rounds so that privacy loss accumulates correctly.
  Epsilon is read after all rounds complete.
  Reported epsilon = max(client epsilons), valid under disjoint partitions
  and image-record-level accounting.
  This does NOT establish patient-level DP.

DS connections
  Unit 3 (Distributed Processing): each client simulates a distributed node
  Unit 5 (Data Stream Mining):      reservoir sampling on client data streams
  Unit 2 (Statistics & Evaluation): epsilon/delta, all classification metrics
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(__file__))

import config
from config import (
    TRAIN_DIR, TEST_DIR, METRICS_CSV, CHECKPOINT_DIR,
    N_CLIENTS, FL_ROUNDS, LOCAL_EPOCHS,
    BATCH_SIZE, LEARNING_RATE, VAL_FRACTION, SEED,
    NOISE_MULTIPLIER, MAX_GRAD_NORM, TARGET_DELTA,
    make_output_dirs,
)
from data_loader import (
    ChestXRayDataset, get_train_transform, get_eval_transform,
    make_train_val_split, partition_for_clients,
)
from model    import build_model
from evaluate import evaluate_model, print_metrics, plot_confusion_matrix, save_result
from reservoir import sample_client_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── FedAvg ────────────────────────────────────────────────────────────────────
def fedavg(global_model: nn.Module, client_updates: list) -> nn.Module:
    """
    Sample-weighted Federated Averaging (McMahan et al. 2017).

    Args:
        global_model   : current global model (used only for structure)
        client_updates : list of (state_dict, n_samples) tuples

    Handles both floating-point parameters (averaged) and integer/bool buffers
    such as BatchNorm running_mean and running_var, which are also averaged,
    and num_batches_tracked, which is summed (then divided, as an approximation).

    Raises ValueError if:
      - No updates are supplied.
      - A client state-dict has different keys from the global model.
      - A client reports zero or negative samples.
    """
    if not client_updates:
        raise ValueError("fedavg: no client updates supplied.")

    global_keys = set(global_model.state_dict().keys())
    for i, (sd, n) in enumerate(client_updates):
        if set(sd.keys()) != global_keys:
            missing   = global_keys - set(sd.keys())
            extra     = set(sd.keys()) - global_keys
            raise ValueError(
                f"fedavg: client {i} state-dict key mismatch. "
                f"Missing: {missing}. Extra: {extra}."
            )
        if n <= 0:
            raise ValueError(
                f"fedavg: client {i} reports {n} samples (must be > 0)."
            )

    total_samples = sum(n for _, n in client_updates)
    new_state     = {}

    for key in global_keys:
        # Weighted average in float32 regardless of original dtype
        agg = torch.zeros_like(
            client_updates[0][0][key], dtype=torch.float32
        )
        for sd, n in client_updates:
            agg += sd[key].float() * (n / total_samples)

        # Cast back to the original dtype
        orig_dtype = client_updates[0][0][key].dtype
        new_state[key] = agg.to(orig_dtype)

    new_model = copy.deepcopy(global_model)
    new_model.load_state_dict(new_state)
    return new_model


# ── Local training — no DP ────────────────────────────────────────────────────
def local_train(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epochs:    int = LOCAL_EPOCHS,
) -> tuple:
    """
    Standard SGD local training for E2.
    Backbone features are kept in eval() so frozen BatchNorm running
    statistics are not updated.

    Returns:
        (trained plain nn.Module, n_local_samples: int)
    """
    local_model = copy.deepcopy(model).to(device)
    optimizer   = Adam(
        (p for p in local_model.parameters() if p.requires_grad),
        lr=LEARNING_RATE,
    )

    for _ in range(epochs):
        local_model.train()
        local_model.features.eval()   # keep frozen BN stats stable
        for images, labels in loader:
            if labels.numel() == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(images), labels)
            loss.backward()
            optimizer.step()

    return local_model, len(loader.dataset)


# ── Local training — with DP ──────────────────────────────────────────────────
def local_train_dp(
    model:         nn.Module,
    loader:        DataLoader,
    criterion:     nn.Module,
    device:        torch.device,
    privacy_engine,             # pre-created PrivacyEngine, reused across rounds
    epochs:        int   = LOCAL_EPOCHS,
    noise_mult:    float = NOISE_MULTIPLIER,
    max_grad_norm: float = MAX_GRAD_NORM,
    target_delta:  float = TARGET_DELTA,
) -> tuple:
    """
    DP-SGD local training using Opacus for E3/E4a/E4b.

    The privacy_engine is created ONCE per client before the FL loop and
    passed in here.  This means accounting accumulates correctly across all
    rounds.  We do NOT call make_private() here; the caller sets it up.

    Returns:
        (trained plain nn.Module, n_local_samples: int, cumulative_epsilon: float)
    """
    # The model passed here is already the Opacus-wrapped version
    local_model = model  # already wrapped by the caller

    local_model.train()
    # Access the underlying module for frozen backbone handling
    try:
        local_model._module.features.eval()
    except AttributeError:
        pass

    optimizer = local_model._dp_optimizer  # already attached by make_private

    for _ in range(epochs):
        local_model.train()
        try:
            local_model._module.features.eval()
        except AttributeError:
            pass
        for images, labels in loader:
            if labels.numel() == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(images), labels)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return local_model._module, len(loader.dataset), epsilon


# ── Federated orchestrator ────────────────────────────────────────────────────
def run_federated(
    experiment_id: str,
    use_dp:        bool  = False,
    sampling_frac: float = 1.0,
    noise_mult:    float = NOISE_MULTIPLIER,
) -> tuple:
    """
    Runs one complete federated experiment (E2, E3, E4a, or E4b).

    DP accounting strategy
    -----------------------
    One PrivacyEngine is created per client before the FL loop starts.
    Each round, make_private() is called on a FRESH copy of the global model
    with the SAME engine so the accountant accumulates privacy loss
    across all rounds.  Epsilon is read once after all rounds finish.
    Reported epsilon = max(client_epsilons), valid under the assumption
    that client partitions are disjoint at the image-record level and
    every client participates in every round.

    Returns:
        (test_metrics: dict, reported_epsilon: float | None)
    """
    make_output_dirs()

    print("\n" + "=" * 55)
    print(f"  EXPERIMENT: {experiment_id}")
    if use_dp:
        print(f"  DP-SGD  noise={noise_mult}  clip={MAX_GRAD_NORM}  δ={TARGET_DELTA}")
        print(f"  DP note: epsilon is IMAGE-RECORD level, not patient level.")
    if sampling_frac < 1.0:
        print(f"  Reservoir sampling fraction = {sampling_frac:.0%}")
    print("=" * 55)
    print(f"  Device     : {DEVICE}")
    print(f"  Clients    : {N_CLIENTS}")
    print(f"  FL rounds  : {FL_ROUNDS}")
    print(f"  Local EP   : {LOCAL_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LEARNING_RATE}")
    print(f"  Seed       : {SEED}")

    # ── 1. Build fixed train/val partitions ───────────────────────────────────
    # Use a no-transform dataset to obtain stable labels for splitting.
    raw_train_ds = ChestXRayDataset(TRAIN_DIR, transform=None)
    all_labels   = raw_train_ds.get_labels()

    # Shared split — identical across ALL federated experiments
    train_idx, val_idx = make_train_val_split(TRAIN_DIR, VAL_FRACTION, SEED)

    # Partition training indices among clients (same partitions for E2/E3/E4)
    base_partitions = partition_for_clients(
        train_idx, all_labels, n_clients=N_CLIENTS, seed=SEED
    )

    # Apply reservoir sampling only for E4a/E4b
    if sampling_frac < 1.0:
        print(f"\n  Reservoir sampling ({sampling_frac:.0%}) per client:")
        client_partitions = [
            sample_client_data(
                indices=part,
                labels=all_labels,
                fraction=sampling_frac,
                seed=SEED + cid,       # independent seed per client
            )
            for cid, part in enumerate(base_partitions)
        ]
    else:
        client_partitions = base_partitions

    # Print final per-client statistics
    print()
    for cid, part in enumerate(client_partitions):
        n_norm = sum(1 for i in part if all_labels[i] == 0)
        n_pneu = sum(1 for i in part if all_labels[i] == 1)
        print(f"  Client {cid + 1}: {len(part):4d} images  "
              f"(NORMAL={n_norm}, PNEUMONIA={n_pneu})")

    total_train_images = sum(len(p) for p in client_partitions)

    # ── 2. DataLoaders ────────────────────────────────────────────────────────
    aug_train_ds = ChestXRayDataset(TRAIN_DIR, transform=get_train_transform())

    def make_client_loader(indices):
        subset = Subset(aug_train_ds, indices)
        # drop_last=True is required by Opacus when not using Poisson sampling;
        # it also avoids a degenerate final batch with size 1.
        return DataLoader(
            subset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0,
            drop_last=(len(indices) > BATCH_SIZE),
        )

    client_loaders = [make_client_loader(p) for p in client_partitions]

    eval_train_ds = ChestXRayDataset(TRAIN_DIR, transform=get_eval_transform())
    val_loader = DataLoader(
        Subset(eval_train_ds, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_ds = ChestXRayDataset(TEST_DIR, transform=get_eval_transform())
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── 3. Build the shared global model (Opacus-compatible) ─────────────────
    # Use freeze_backbone=False so all parameters exist in the state-dict;
    # we then manually freeze features. This ensures E2/E3/E4 all share the
    # same architecture and the same state-dict keys.
    global_model = build_model(num_classes=2, freeze_backbone=False).to(DEVICE)

    if use_dp:
        try:
            from opacus.validators import ModuleValidator
        except ImportError:
            raise ImportError("Install Opacus: pip install opacus")

        if not ModuleValidator.is_valid(global_model):
            global_model = ModuleValidator.fix(global_model)
            print("  Model modified by ModuleValidator.fix() for Opacus compatibility.")

        errors = ModuleValidator.validate(global_model)
        if errors:
            raise RuntimeError(
                f"Model failed Opacus validation after fix(): {errors}"
            )
        print("  Opacus ModuleValidator: OK")

    # Freeze the feature extractor (same for E2/E3/E4)
    for param in global_model.features.parameters():
        param.requires_grad = False
    print(f"  Trainable parameters : "
          f"{sum(p.numel() for p in global_model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss()

    # ── 4. Per-client PrivacyEngine (created ONCE, reused across all rounds) ──
    # This is the key correctness requirement: accounting must accumulate over
    # all steps across all rounds, not reset each round.
    if use_dp:
        try:
            from opacus import PrivacyEngine
        except ImportError:
            raise ImportError("Install Opacus: pip install opacus")

        privacy_engines = [
            PrivacyEngine(accountant="rdp") for _ in range(N_CLIENTS)
        ]
        # For each client, build the initial (round 0) wrapped model/optimizer/loader.
        # We store these and REPLACE the model/optimizer each round but keep the engine.
        client_dp_states = []   # list of [dp_model, dp_optimizer, dp_loader]
        for cid, (engine, loader) in enumerate(
            zip(privacy_engines, client_loaders)
        ):
            init_model = copy.deepcopy(global_model).to(DEVICE)
            init_opt   = Adam(
                (p for p in init_model.parameters() if p.requires_grad),
                lr=LEARNING_RATE,
            )
            dp_model, dp_opt, dp_loader = engine.make_private(
                module=init_model,
                optimizer=init_opt,
                data_loader=loader,
                noise_multiplier=noise_mult,
                max_grad_norm=MAX_GRAD_NORM,
                poisson_sampling=True,
            )
            # Attach dp_optimizer as an attribute so local_train_dp can access it
            dp_model._dp_optimizer = dp_opt
            client_dp_states.append((engine, dp_model, dp_loader))

    # ── 5. Federated loop ─────────────────────────────────────────────────────
    best_val_f1  = float("-inf")
    best_weights = None
    safe_name    = (
        experiment_id.lower()
        .replace(" ", "_").replace("-", "").replace("/", "").replace("+", "")
    )
    checkpoint_path = str(config.CHECKPOINT_DIR / f"{safe_name}_best.pt")

    start_time = time.time()

    for rnd in range(1, FL_ROUNDS + 1):
        print(f"\n  ── Round {rnd}/{FL_ROUNDS} ──")
        client_updates = []

        for cid, loader in enumerate(client_loaders):
            if len(loader.dataset) == 0:
                print(f"    Client {cid + 1}: skipped (empty partition)")
                continue

            if use_dp:
                engine, dp_model, dp_loader = client_dp_states[cid]

                # Load latest global weights into the wrapped model's underlying module
                dp_model._module.load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                dp_model._module.features.eval()

                plain_model, n_samples, eps = local_train_dp(
                    model=dp_model,
                    loader=dp_loader,
                    criterion=criterion,
                    device=DEVICE,
                    privacy_engine=engine,
                    epochs=LOCAL_EPOCHS,
                    noise_mult=noise_mult,
                    max_grad_norm=MAX_GRAD_NORM,
                    target_delta=TARGET_DELTA,
                )
                # plain_model is dp_model._module — use its state_dict
                client_updates.append((
                    copy.deepcopy(plain_model.state_dict()), n_samples
                ))
                print(f"    Client {cid + 1}: {n_samples} samples  "
                      f"cumul. ε={eps:.4f}")

            else:
                local_model, n_samples = local_train(
                    global_model, loader, criterion, DEVICE
                )
                client_updates.append((local_model.state_dict(), n_samples))
                print(f"    Client {cid + 1}: {n_samples} samples")

        if not client_updates:
            raise RuntimeError("All clients were skipped — no updates to aggregate.")

        # FedAvg
        global_model = fedavg(global_model, client_updates).to(DEVICE)

        # Validation (never use test set here)
        val_metrics = evaluate_model(global_model, val_loader, DEVICE)
        print(f"  Round {rnd} val — "
              f"acc={val_metrics['accuracy']:.4f}  "
              f"recall={val_metrics['recall']:.4f}  "
              f"f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_weights = copy.deepcopy(global_model.state_dict())
            torch.save(best_weights, checkpoint_path)
            print(f"  New best (val F1={best_val_f1:.4f}) saved → {checkpoint_path}")

    training_time = time.time() - start_time

    if best_weights is None:
        raise RuntimeError(
            "No checkpoint saved — all validation scores were -inf."
        )
    global_model.load_state_dict(best_weights)

    # ── 6. Test evaluation (once, on the held-out test set) ───────────────────
    test_metrics = evaluate_model(global_model, test_loader, DEVICE)
    print_metrics(test_metrics, f"{experiment_id} — Test Set Results")
    plot_confusion_matrix(
        test_metrics,
        title=experiment_id,
        save_name=f"cm_{safe_name}.png",
    )

    # ── 7. Privacy budget reporting ───────────────────────────────────────────
    client_epsilons: dict = {}
    if use_dp:
        for cid, (engine, _, _) in enumerate(client_dp_states):
            eps = engine.get_epsilon(delta=TARGET_DELTA)
            client_epsilons[cid + 1] = round(eps, 4)
        reported_eps = round(max(client_epsilons.values()), 4)
        print(f"\n  Per-client cumulative ε : {client_epsilons}")
        print(f"  Overall ε bound (max)   : {reported_eps:.4f}")
        print(f"  δ                       : {TARGET_DELTA}")
        print(f"  Assumption              : disjoint image-record partitions, "
              f"Poisson sampling, {N_CLIENTS} clients each in every round.")
    else:
        reported_eps = None

    # ── 8. Save result row ────────────────────────────────────────────────────
    row = {
        "experiment":         experiment_id,
        "n_clients":          N_CLIENTS,
        "fl_rounds":          FL_ROUNDS,
        "local_epochs":       LOCAL_EPOCHS,
        "batch_size":         BATCH_SIZE,
        "learning_rate":      LEARNING_RATE,
        "dp_enabled":         use_dp,
        "noise_multiplier":   noise_mult if use_dp else None,
        "max_grad_norm":      MAX_GRAD_NORM if use_dp else None,
        "target_delta":       TARGET_DELTA if use_dp else None,
        "epsilon":            reported_eps,
        "client_epsilons":    json.dumps(client_epsilons) if use_dp else None,
        "privacy_assumption": (
            "image-record level; disjoint partitions; max-client epsilon"
        ) if use_dp else None,
        "sampling_frac":      sampling_frac,
        "n_train_images":     total_train_images,
        "training_time_s":    round(training_time, 1),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in test_metrics.items()},
    }
    save_result(row, METRICS_CSV)

    print(f"\n  Training time : {training_time:.1f}s")
    print(f"  {experiment_id} complete.\n")
    return test_metrics, reported_eps


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import config as cfg
    cfg.verify_paths()

    run_federated("E2 - Federated No DP",
                  use_dp=False, sampling_frac=1.0)

    run_federated("E3 - Federated + DP",
                  use_dp=True, sampling_frac=1.0)

    run_federated("E4a - FL+DP+RS50%",
                  use_dp=True, sampling_frac=0.50)

    run_federated("E4b - FL+DP+RS25%",
                  use_dp=True, sampling_frac=0.25)

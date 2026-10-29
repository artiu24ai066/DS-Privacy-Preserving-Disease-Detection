"""
federated.py
------------
Experiments E2, E3, E4a, E4b — Federated Learning simulation.

E2  : Federated learning, no differential privacy
E3  : Federated learning + DP-SGD (Opacus)
E4a : Federated + DP-SGD + Reservoir Sampling at 50%
E4b : Federated + DP-SGD + Reservoir Sampling at 25%

Architecture
  All experiments (E1–E4b) use the same DP-compatible MobileNetV2.
  ModuleValidator.fix() is called inside build_model(), replacing BatchNorm
  with GroupNorm throughout the backbone.  This is applied to E1 as well
  so architectures are identical and results are comparable.
  BatchNorm running statistics are absent from the state dict after fix().

Frozen backbone
  model.features.parameters() all have requires_grad=False.
  During local training, features.eval() is called to prevent any
  incidental stat updates (GroupNorm has no running stats, but the call
  is safe and makes the intent explicit).

FedAvg
  Sample-weighted aggregation (McMahan et al. 2017).
  All tensor types (float and integer) are averaged in float32 then
  cast back to their original dtype.  This handles GroupNorm's weight
  and bias buffers correctly.

DP accounting (E3/E4a/E4b)
  One PrivacyEngine per client, created ONCE before the FL loop.
  The same engine is reused across all rounds so the RDP accountant
  accumulates steps from every round without resetting.

  Each round: global weights are loaded into dp_model._module (the
  underlying plain module inside the GradSampleModule wrapper).
  The wrapped optimizer (dp_opt) is reused; its parameter references
  remain valid because _module is never replaced, only its weights
  are updated.

  Epsilon is read from the engine after all rounds complete.
  Reported epsilon = max(client epsilons) — valid as an overall bound
  under the assumption that:
    1. Client partitions are disjoint at the image-record level.
    2. Every client participates in every round.
    3. Poisson sampling is used within each client's DataLoader.
    4. The DP guarantee is at the IMAGE-RECORD level, NOT patient level.
       A single patient may appear in multiple images; that is not
       accounted for here.

DS connections
  Unit 3 (Distributed Processing): clients simulate distributed nodes
  Unit 5 (Data Stream Mining):      reservoir sampling on client data
  Unit 2 (Statistics & Evaluation): epsilon/delta, all metrics
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
    make_train_val_split, partition_for_clients, get_class_weights,
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
        global_model   : current global model (used for structure and validation)
        client_updates : list of (state_dict, n_samples) tuples

    All tensor dtypes are averaged in float32 then cast back to the original
    dtype.  This handles both floating-point parameters (features, classifier)
    and any remaining integer/boolean buffers.

    Raises ValueError if:
      - No updates are supplied.
      - Any client state-dict has different keys from the global model.
      - Any client reports zero or negative sample count.
      - Any tensor shape does not match between client and global model.
    """
    if not client_updates:
        raise ValueError("fedavg: no client updates supplied.")

    global_sd   = global_model.state_dict()
    global_keys = set(global_sd.keys())

    for i, (sd, n) in enumerate(client_updates):
        if set(sd.keys()) != global_keys:
            missing = global_keys - set(sd.keys())
            extra   = set(sd.keys()) - global_keys
            raise ValueError(
                f"fedavg: client {i} state-dict key mismatch. "
                f"Missing: {missing}. Extra: {extra}."
            )
        if n <= 0:
            raise ValueError(
                f"fedavg: client {i} reports {n} samples (must be > 0)."
            )
        for key in global_keys:
            if sd[key].shape != global_sd[key].shape:
                raise ValueError(
                    f"fedavg: client {i} key '{key}' shape {sd[key].shape} "
                    f"!= global {global_sd[key].shape}."
                )

    total_samples = sum(n for _, n in client_updates)
    new_state     = {}

    for key in global_keys:
        orig_dtype = global_sd[key].dtype
        agg = torch.zeros_like(global_sd[key], dtype=torch.float32)
        for sd, n in client_updates:
            agg += sd[key].float() * (n / total_samples)
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
    Backbone features are kept in eval() so frozen GroupNorm weight/bias
    do not accidentally receive gradient updates.

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
        local_model.features.eval()
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
    dp_model:      nn.Module,       # GradSampleModule (already wrapped)
    dp_opt,                         # DPOptimizer (already created by make_private)
    dp_loader:     DataLoader,      # Poisson-sampled loader (from make_private)
    criterion:     nn.Module,
    privacy_engine,                 # PrivacyEngine — reused across rounds for accounting
    device:        torch.device,
    epochs:        int   = LOCAL_EPOCHS,
    target_delta:  float = TARGET_DELTA,
) -> tuple:
    """
    DP-SGD local training for E3/E4a/E4b.

    The dp_model, dp_opt, and dp_loader were created by PrivacyEngine.make_private()
    ONCE before the FL loop and are reused each round.  Between rounds, the caller
    loads the latest global weights into dp_model._module so the model starts each
    round from the aggregated global state.

    The PrivacyEngine's RDP accountant accumulates steps across all rounds —
    this is the correct design for measuring total privacy expenditure.

    Returns:
        (plain nn.Module (dp_model._module), n_local_samples: int,
         cumulative_epsilon: float)
    """
    dp_model.train()
    # GroupNorm has no running statistics, but calling .eval() on features is
    # safe and makes clear that frozen backbone params are not being updated.
    try:
        dp_model._module.features.eval()
    except AttributeError:
        pass

    for _ in range(epochs):
        dp_model.train()
        try:
            dp_model._module.features.eval()
        except AttributeError:
            pass
        for images, labels in dp_loader:
            if labels.numel() == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            dp_opt.zero_grad()
            loss = criterion(dp_model(images), labels)
            loss.backward()
            dp_opt.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    # dp_model._module is the plain underlying nn.Module (GradSampleModule wraps it)
    return dp_model._module, len(dp_loader.dataset), epsilon


# ── Federated orchestrator ────────────────────────────────────────────────────
def run_federated(
    experiment_id: str,
    use_dp:        bool  = False,
    sampling_frac: float = 1.0,
    noise_mult:    float = NOISE_MULTIPLIER,
) -> tuple:
    """
    Runs one complete federated experiment (E2, E3, E4a, or E4b).

    DP design summary
    -----------------
    Each client has one PrivacyEngine created before the loop.
    make_private() is called once per client; it wraps the client's local
    model copy into a GradSampleModule (dp_model) and creates a DPOptimizer
    (dp_opt).  Both are reused across all FL rounds.

    Each round:
      1. Load current global weights: dp_model._module.load_state_dict(global)
      2. Call local_train_dp() — trains for LOCAL_EPOCHS using dp_loader
      3. Extract updated weights: dp_model._module.state_dict()
      4. Aggregate via FedAvg

    The accountant step count increases with every optimizer.step() call
    across all rounds, so epsilon is the TOTAL privacy cost over all rounds.

    Returns:
        (test_metrics: dict, reported_epsilon: float | None)
    """
    make_output_dirs()

    print("\n" + "=" * 55)
    print(f"  EXPERIMENT: {experiment_id}")
    if use_dp:
        print(f"  DP-SGD  noise={noise_mult}  clip={MAX_GRAD_NORM}  δ={TARGET_DELTA}")
        print("  NOTE: epsilon is IMAGE-RECORD level, NOT patient level.")
        print("  One patient may have multiple records; this is not accounted for.")
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

    # ── 1. Build fixed train/val/client partitions ────────────────────────────
    raw_ds     = ChestXRayDataset(TRAIN_DIR, transform=None)
    all_labels = raw_ds.get_labels()

    # These are identical across E2/E3/E4a/E4b — same seed, same split
    train_idx, val_idx = make_train_val_split(TRAIN_DIR, VAL_FRACTION, SEED)
    base_partitions    = partition_for_clients(
        train_idx, all_labels, n_clients=N_CLIENTS, seed=SEED
    )

    # Reservoir sampling for E4a/E4b only; E3 uses full base_partitions
    if sampling_frac < 1.0:
        print(f"\n  Reservoir sampling ({sampling_frac:.0%}) per client:")
        client_partitions = [
            sample_client_data(
                indices=part,
                labels=all_labels,
                fraction=sampling_frac,
                seed=SEED + cid,
            )
            for cid, part in enumerate(base_partitions)
        ]
    else:
        client_partitions = base_partitions

    print()
    for cid, part in enumerate(client_partitions):
        n_norm = sum(1 for i in part if all_labels[i] == 0)
        n_pneu = sum(1 for i in part if all_labels[i] == 1)
        print(f"  Client {cid + 1}: {len(part):4d} images  "
              f"(NORMAL={n_norm}, PNEUMONIA={n_pneu})")

    total_train_images = sum(len(p) for p in client_partitions)

    # ── 2. DataLoaders ────────────────────────────────────────────────────────
    aug_ds = ChestXRayDataset(TRAIN_DIR, transform=get_train_transform())

    def make_client_loader(indices, is_dp_loader=False):
        subset = Subset(aug_ds, indices)
        # drop_last=True only for non-DP loaders where batch size matters
        # for training stability (avoids single-sample batches).
        # For DP experiments, Opacus replaces this loader with a Poisson
        # sampler anyway, so drop_last here doesn't affect DP training.
        # For E2 (no DP), we keep drop_last=False to use all training data.
        return DataLoader(
            subset, batch_size=BATCH_SIZE,
            shuffle=True, num_workers=0,
            drop_last=False,   # never silently discard training images
        )

    client_loaders = [make_client_loader(p) for p in client_partitions]

    eval_ds    = ChestXRayDataset(TRAIN_DIR, transform=get_eval_transform())
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_ds     = ChestXRayDataset(TEST_DIR, transform=get_eval_transform())
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── 3. Global model ───────────────────────────────────────────────────────
    # freeze_backbone=False so all parameters appear in the state dict;
    # backbone is manually frozen below.  This ensures FedAvg can aggregate
    # the full state dict (all 158 keys after fix()).
    global_model = build_model(num_classes=2, freeze_backbone=False).to(DEVICE)
    for param in global_model.features.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"\n  Trainable parameters : {trainable:,}")
    print(f"  State-dict keys      : {len(global_model.state_dict())}")

    # Class-weighted loss to counteract the 74.3% PNEUMONIA imbalance.
    # Without weights, CE loss is minimised by predicting the majority class
    # for all images, collapsing specificity to 0.
    class_weights = get_class_weights(TRAIN_DIR, train_idx, all_labels, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── 4. Per-client DP setup (created ONCE, reused across rounds) ───────────
    if use_dp:
        try:
            from opacus import PrivacyEngine
        except ImportError:
            raise ImportError("Install Opacus: pip install opacus")

        client_dp_states = []  # list of (engine, dp_model, dp_opt, dp_loader)

        for cid, base_loader in enumerate(client_loaders):
            engine    = PrivacyEngine(accountant="rdp")
            # Each client gets its own model copy — weights will be synced each round
            cli_model = copy.deepcopy(global_model).to(DEVICE)
            cli_opt   = Adam(
                (p for p in cli_model.parameters() if p.requires_grad),
                lr=LEARNING_RATE,
            )
            dp_model_c, dp_opt_c, dp_loader_c = engine.make_private(
                module=cli_model,
                optimizer=cli_opt,
                data_loader=base_loader,
                noise_multiplier=noise_mult,
                max_grad_norm=MAX_GRAD_NORM,
                poisson_sampling=True,
            )
            # Verify _module is accessible (confirmed for Opacus 1.6.0)
            assert hasattr(dp_model_c, "_module"), (
                "dp_model does not have _module attribute — "
                "Opacus API may have changed. Cannot sync global weights."
            )
            client_dp_states.append((engine, dp_model_c, dp_opt_c, dp_loader_c))

        print(f"  DP engines created   : {len(client_dp_states)}")
        print(f"  Poisson sampling     : True")
        print(f"  Noise multiplier     : {noise_mult}")
        print(f"  Max grad norm        : {MAX_GRAD_NORM}")
        print(f"  Target delta         : {TARGET_DELTA}")

    # ── 5. Federated loop ─────────────────────────────────────────────────────
    best_val_f1  = float("-inf")
    best_weights = None
    safe_name    = (
        experiment_id.lower()
        .replace(" ", "_").replace("-", "").replace("/", "").replace("+", "")
    )
    checkpoint_path = str(config.CHECKPOINT_DIR / f"{safe_name}_best.pt")
    start_time      = time.time()

    for rnd in range(1, FL_ROUNDS + 1):
        print(f"\n  ── Round {rnd}/{FL_ROUNDS} ──")
        client_updates = []

        for cid, base_loader in enumerate(client_loaders):
            if len(base_loader.dataset) == 0:
                print(f"    Client {cid + 1}: skipped (empty partition)")
                continue

            if use_dp:
                engine, dp_model_c, dp_opt_c, dp_loader_c = client_dp_states[cid]

                # Sync global weights into the client's wrapped model.
                # dp_model._module holds the underlying plain nn.Module;
                # its state-dict keys match the global model's keys.
                dp_model_c._module.load_state_dict(
                    copy.deepcopy(global_model.state_dict())
                )
                dp_model_c._module.features.eval()

                plain_model, n_samples, eps = local_train_dp(
                    dp_model=dp_model_c,
                    dp_opt=dp_opt_c,
                    dp_loader=dp_loader_c,
                    criterion=criterion,
                    privacy_engine=engine,
                    device=DEVICE,
                    epochs=LOCAL_EPOCHS,
                    target_delta=TARGET_DELTA,
                )
                client_updates.append((
                    copy.deepcopy(plain_model.state_dict()), n_samples
                ))
                print(f"    Client {cid + 1}: {n_samples} samples  cumul. ε={eps:.4f}")

            else:
                local_model, n_samples = local_train(
                    global_model, base_loader, criterion, DEVICE
                )
                client_updates.append((local_model.state_dict(), n_samples))
                print(f"    Client {cid + 1}: {n_samples} samples")

        if not client_updates:
            raise RuntimeError(
                f"Round {rnd}: all clients skipped — nothing to aggregate."
            )

        global_model = fedavg(global_model, client_updates).to(DEVICE)

        # Validate on val split — never touch the test set here
        val_metrics = evaluate_model(global_model, val_loader, DEVICE)
        print(f"  Round {rnd} val — "
              f"acc={val_metrics['accuracy']:.4f}  "
              f"recall={val_metrics['recall']:.4f}  "
              f"f1={val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_weights = copy.deepcopy(global_model.state_dict())
            torch.save(best_weights, checkpoint_path)
            print(f"  New best (val F1={best_val_f1:.4f}) → {checkpoint_path}")

    training_time = time.time() - start_time

    if best_weights is None:
        raise RuntimeError(
            "No checkpoint saved — federated loop completed without "
            "any validation improvement."
        )
    global_model.load_state_dict(best_weights)

    # ── 6. Final test evaluation (once, on held-out test set) ─────────────────
    test_metrics = evaluate_model(global_model, test_loader, DEVICE)
    print_metrics(test_metrics, f"{experiment_id} — Test Set Results")
    plot_confusion_matrix(
        test_metrics,
        title=experiment_id,
        save_name=f"cm_{safe_name}.png",
    )

    # ── 7. Privacy budget ─────────────────────────────────────────────────────
    client_epsilons: dict = {}
    if use_dp:
        for cid, (engine, _, _, _) in enumerate(client_dp_states):
            eps = engine.get_epsilon(delta=TARGET_DELTA)
            client_epsilons[cid + 1] = round(eps, 4)
        reported_eps = round(max(client_epsilons.values()), 4)
        print(f"\n  Per-client cumulative ε : {client_epsilons}")
        print(f"  Overall ε (max-client)  : {reported_eps:.4f}")
        print(f"  δ                       : {TARGET_DELTA}")
        print(f"  Adjacency               : image-record level (one record differs)")
        print(f"  Assumption              : disjoint partitions, Poisson sampling,")
        print(f"                            all clients in every round")
        print(f"  NOT covered             : patient-level privacy (multiple images")
        print(f"                            per patient are not protected together)")
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
            "image-record; disjoint partitions; Poisson; all clients every round; "
            "max-client epsilon; NOT patient-level"
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

    run_federated("E2 - Federated No DP",  use_dp=False, sampling_frac=1.0)
    run_federated("E3 - Federated + DP",   use_dp=True,  sampling_frac=1.0)
    run_federated("E4a - FL+DP+RS50%",     use_dp=True,  sampling_frac=0.50)
    run_federated("E4b - FL+DP+RS25%",     use_dp=True,  sampling_frac=0.25)

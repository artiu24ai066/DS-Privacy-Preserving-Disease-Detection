"""
config.py
---------
Single source of truth for all dataset paths and shared hyperparameters.
Every experiment module imports from here so paths are never defined
in more than one place.

Dataset layout (verified against actual directory):
  chest_xray/
    train/
      NORMAL/     (1341 images)
      PNEUMONIA/  (3875 images)
    test/
      NORMAL/     (234 images)
      PNEUMONIA/  (390 images)
    val/
      NORMAL/     (8 images — too small; we build our own split from train/)
      PNEUMONIA/  (8 images — too small; we build our own split from train/)

NOTE: there is also a chest_xray/chest_xray/ nested copy from a Mac ZIP
extraction.  That copy does NOT contain images and is not used.
"""

import os
from pathlib import Path

# ── Project root (one level above src/) ───────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset paths ──────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "chest_xray"
TRAIN_DIR = str(DATA_DIR / "train")
TEST_DIR  = str(DATA_DIR / "test")
# Official val/ is retained for reference only — too small to use directly.
# All experiments build a 20% stratified validation split from TRAIN_DIR.
OFFICIAL_VAL_DIR = str(DATA_DIR / "val")

# ── Output paths ───────────────────────────────────────────────────────────────
RESULTS_DIR    = PROJECT_ROOT / "results"
PLOTS_DIR      = RESULTS_DIR / "plots"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
METRICS_CSV    = str(RESULTS_DIR / "metrics.csv")

# ── Shared hyperparameters (used by E1–E4b) ────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
VAL_FRACTION  = 0.20
SEED          = 42

# ── Federated hyperparameters ──────────────────────────────────────────────────
N_CLIENTS    = 3
FL_ROUNDS    = 3
LOCAL_EPOCHS = 2

# ── DP-SGD hyperparameters ─────────────────────────────────────────────────────
# These apply to E3, E4a, E4b.
# Privacy guarantee: (ε, δ)-DP at the image-record level under the assumption
# that client partitions are disjoint, clients participate in every round,
# and Poisson sampling is used within each client's local DataLoader.
# The reported ε is the maximum cumulative epsilon across clients.
# This does NOT guarantee patient-level DP (a patient may have multiple images).
NOISE_MULTIPLIER = 1.0   # higher → more privacy, lower accuracy
MAX_GRAD_NORM    = 1.0   # per-sample gradient clipping norm
TARGET_DELTA     = 1e-5  # δ parameter; set < 1/n_train_images


def verify_paths() -> bool:
    """
    Checks that TRAIN_DIR and TEST_DIR exist and contain images.
    Prints a clear error message if not.
    Returns True if everything looks correct.
    """
    ok = True
    for name, directory in [("TRAIN_DIR", TRAIN_DIR), ("TEST_DIR", TEST_DIR)]:
        for cls in ("NORMAL", "PNEUMONIA"):
            p = Path(directory) / cls
            if not p.exists():
                print(f"[PATH ERROR] {name}/{cls} not found: {p}")
                ok = False
            else:
                n = len(list(p.glob("*.jpeg"))) + len(list(p.glob("*.jpg")))
                if n == 0:
                    print(f"[PATH ERROR] {name}/{cls} exists but contains no images: {p}")
                    ok = False
    if ok:
        print("[config] Dataset paths verified OK.")
    return ok


def make_output_dirs():
    """Creates all output directories needed before writing results."""
    for d in (RESULTS_DIR, PLOTS_DIR, CHECKPOINT_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

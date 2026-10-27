"""
run_all.py
----------
Master script — runs all five experiments in order and then generates
all visualizations.

    E1  — Centralized baseline
    E2  — Federated learning, no DP
    E3  — Federated learning + DP-SGD
    E4a — Federated + DP + Reservoir Sampling (50%)
    E4b — Federated + DP + Reservoir Sampling (25%)

Usage
-----
    # Full run (may take 30-90 min on CPU)
    python src/run_all.py

    # Quick smoke-test (1 epoch / 1 round — NOT for final results)
    python src/run_all.py --quick

QUICK_TEST mode reduces epochs and rounds to 1 so the code path can be
verified quickly.  Results from quick mode are labelled as such and
must NOT be used as final experimental results.
"""

import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import config
from config import METRICS_CSV, make_output_dirs

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Run all experiments.")
parser.add_argument(
    "--quick", action="store_true",
    help="Smoke-test mode: 1 epoch / 1 round. NOT suitable for final results."
)
args, _ = parser.parse_known_args()
QUICK_TEST: bool = args.quick

if QUICK_TEST:
    print("\n" + "!" * 55)
    print("  QUICK_TEST mode is ON.")
    print("  Epochs = 1, FL rounds = 1, local epochs = 1.")
    print("  These results are NOT valid for final reporting.")
    print("!" * 55 + "\n")


def apply_quick_test_overrides():
    """Reduce iterations so a full code-path check completes quickly."""
    import train_centralized
    import federated
    train_centralized.EPOCHS       = 1
    federated.FL_ROUNDS            = 1
    federated.LOCAL_EPOCHS         = 1


def archive_previous_results():
    """
    Move metrics.csv to a timestamped archive before starting a full run.
    Individual experiment results are preserved; they are not deleted.
    """
    metrics_path = Path(METRICS_CSV)
    if not metrics_path.exists():
        return
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = metrics_path.parent / f"metrics_{timestamp}.csv"
    shutil.move(str(metrics_path), str(archive_path))
    print(f"  Previous results archived → {archive_path}")


def run_experiment(label, fn, *args, **kwargs):
    """
    Runs fn(*args, **kwargs), catches exceptions, and returns
    (success: bool, result).
    """
    try:
        result = fn(*args, **kwargs)
        print(f"  [{label}] finished OK")
        return True, result
    except Exception:
        print(f"\n  [{label}] FAILED:")
        traceback.print_exc()
        return False, None


def main():
    make_output_dirs()
    config.verify_paths()

    if not QUICK_TEST:
        archive_previous_results()

    if QUICK_TEST:
        apply_quick_test_overrides()

    mode_label = "QUICK-TEST" if QUICK_TEST else "FULL RUN"
    print("\n" + "█" * 55)
    print(f"  Privacy-Preserving Pneumonia Detection — {mode_label}")
    print("█" * 55)

    from train_centralized import run as run_e1
    from federated import run_federated

    experiments = [
        ("E1", run_e1,                          {}),
        ("E2", run_federated, {"experiment_id": "E2 - Federated No DP",
                               "use_dp": False,  "sampling_frac": 1.0}),
        ("E3", run_federated, {"experiment_id": "E3 - Federated + DP",
                               "use_dp": True,   "sampling_frac": 1.0}),
        ("E4a", run_federated, {"experiment_id": "E4a - FL+DP+RS50%",
                                "use_dp": True,   "sampling_frac": 0.50}),
        ("E4b", run_federated, {"experiment_id": "E4b - FL+DP+RS25%",
                                "use_dp": True,   "sampling_frac": 0.25}),
    ]

    failed = []
    for label, fn, kwargs in experiments:
        ok, _ = run_experiment(label, fn, **kwargs)
        if not ok:
            failed.append(label)

    # ── Visualizations (only if at least some experiments succeeded) ──────────
    metrics_path = Path(METRICS_CSV)
    if metrics_path.exists():
        print("\n" + "=" * 55)
        print("  Generating visualizations...")
        print("=" * 55)
        try:
            from visualize import (
                load_results, print_summary_table,
                plot_comparison, plot_training_time,
                plot_reservoir_effect, plot_privacy_vs_performance,
                plot_recall_focus, plot_class_distribution,
            )
            plot_class_distribution(config.TRAIN_DIR, config.TEST_DIR)
            df = load_results()
            print_summary_table(df)
            plot_comparison(df)
            plot_training_time(df)
            plot_reservoir_effect(df)
            plot_privacy_vs_performance(df)
            plot_recall_focus(df)
            print("  All plots saved → results/plots/")
        except Exception:
            print("  Visualization failed:")
            traceback.print_exc()
    else:
        print("  No metrics.csv found — skipping visualization.")

    # ── Final status ──────────────────────────────────────────────────────────
    print("\n" + "█" * 55)
    if failed:
        print(f"  COMPLETED WITH FAILURES: {', '.join(failed)}")
        print("  Check the error output above for details.")
    else:
        print(f"  ALL EXPERIMENTS COMPLETE ({mode_label})")
    if QUICK_TEST:
        print("  ⚠ Results are from QUICK_TEST mode — not valid for reporting.")
    print("  Results → results/metrics.csv")
    print("  Plots   → results/plots/")
    print("█" * 55 + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

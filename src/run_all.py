"""
run_all.py
----------
Master script that runs all experiments in order:

  E1  — Centralized baseline
  E2  — Federated learning, no DP
  E3  — Federated learning + DP-SGD
  E4a — Federated + DP + Reservoir 50%
  E4b — Federated + DP + Reservoir 25%

Then generates all visualizations.

Usage:
    python src/run_all.py

You can also run individual experiments by importing from their modules.

IMPORTANT: This script may take 30–90 minutes on CPU depending on your hardware.
For a quick test, reduce EPOCHS and FL_ROUNDS in the config at the top of this file.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Quick-test override (set to True to run a fast smoke-test) ─────────────────
QUICK_TEST = True

if QUICK_TEST:
    import train_centralized
    import federated
    train_centralized.EPOCHS       = 1
    federated.FL_ROUNDS            = 1
    federated.LOCAL_EPOCHS         = 1
    print("⚠ QUICK_TEST mode: 1 epoch / 1 round only")


def main():
    print("\n" + "█"*55)
    print("  Privacy-Preserving Pneumonia Detection")
    print("  Running all experiments...")
    print("█"*55)

    # ── E1: Centralized baseline ───────────────────────────────────────────────
    from train_centralized import run as run_e1
    run_e1()

    # ── E2: Federated, no DP ──────────────────────────────────────────────────
    from federated import run_federated
    run_federated(
        experiment_id="E2 - Federated No DP",
        use_dp=False,
        sampling_frac=1.0,
    )

    # ── E3: Federated + DP-SGD ────────────────────────────────────────────────
    run_federated(
        experiment_id="E3 - Federated + DP",
        use_dp=True,
        sampling_frac=1.0,
    )

    # ── E4a: Federated + DP + Reservoir 50% ──────────────────────────────────
    run_federated(
        experiment_id="E4a - FL+DP+RS50%",
        use_dp=True,
        sampling_frac=0.50,
    )

    # ── E4b: Federated + DP + Reservoir 25% ──────────────────────────────────
    run_federated(
        experiment_id="E4b - FL+DP+RS25%",
        use_dp=True,
        sampling_frac=0.25,
    )

    # ── Generate all plots ────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Generating visualizations...")
    print("="*55)
    from visualize import (
        load_results, print_summary_table,
        plot_comparison, plot_training_time,
        plot_reservoir_effect, plot_privacy_vs_performance,
        plot_recall_focus,
    )

    BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
    TRAIN_DIR = os.path.join(BASE_DIR, "chest_xray", "train")
    TEST_DIR  = os.path.join(BASE_DIR, "chest_xray", "chest_xray", "test")

    from visualize import plot_class_distribution
    plot_class_distribution(TRAIN_DIR, TEST_DIR)

    df = load_results()
    print_summary_table(df)
    plot_comparison(df)
    plot_training_time(df)
    plot_reservoir_effect(df)
    plot_privacy_vs_performance(df)
    plot_recall_focus(df)

    print("\n" + "█"*55)
    print("  All experiments complete!")
    print("  Results → results/metrics.csv")
    print("  Plots   → results/plots/")
    print("█"*55 + "\n")


if __name__ == "__main__":
    main()

"""
tests/test_project.py
---------------------
Automated tests for the Privacy-Preserving Pneumonia Detection project.

Run with:
    python -m pytest tests/ -v

Tests are grouped by component and use synthetic data where possible
so they do not require the full dataset or a GPU.

Where a test requires the real dataset it is marked with
@pytest.mark.requires_data and will be skipped automatically when
the dataset is not found.
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Make src importable ───────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config
from config import TRAIN_DIR, TEST_DIR

# ── Marker: skip if real dataset absent ──────────────────────────────────────
DATA_AVAILABLE = (
    Path(TRAIN_DIR).exists()
    and (Path(TRAIN_DIR) / "NORMAL").exists()
    and (Path(TRAIN_DIR) / "PNEUMONIA").exists()
)
requires_data = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason="Real dataset not found at configured TRAIN_DIR",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG & PATHS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_project_root_exists(self):
        assert config.PROJECT_ROOT.exists()

    def test_paths_are_strings(self):
        assert isinstance(config.TRAIN_DIR, str)
        assert isinstance(config.TEST_DIR, str)
        assert isinstance(config.METRICS_CSV, str)

    def test_output_paths_are_path_objects(self):
        assert isinstance(config.RESULTS_DIR, Path)
        assert isinstance(config.PLOTS_DIR, Path)
        assert isinstance(config.CHECKPOINT_DIR, Path)

    def test_make_output_dirs(self):
        config.make_output_dirs()
        assert config.RESULTS_DIR.exists()
        assert config.PLOTS_DIR.exists()
        assert config.CHECKPOINT_DIR.exists()

    @requires_data
    def test_verify_paths_passes(self):
        assert config.verify_paths() is True

    def test_hyperparameters_positive(self):
        assert config.BATCH_SIZE > 0
        assert config.LEARNING_RATE > 0
        assert 0 < config.VAL_FRACTION < 1
        assert config.N_CLIENTS >= 2
        assert config.FL_ROUNDS >= 1
        assert config.LOCAL_EPOCHS >= 1
        assert config.NOISE_MULTIPLIER > 0
        assert config.MAX_GRAD_NORM > 0
        assert 0 < config.TARGET_DELTA < 1


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RESERVOIR SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

class TestReservoir:
    def setup_method(self):
        from reservoir import reservoir_sample, sample_client_data
        self.rs   = reservoir_sample
        self.scd  = sample_client_data

    def test_basic_output_size(self):
        result = self.rs(list(range(100)), k=20, seed=42)
        assert len(result) == 20

    def test_full_stream_when_k_ge_n(self):
        stream = [1, 2, 3]
        result = self.rs(stream, k=10, seed=0)
        assert sorted(result) == stream

    def test_exact_k_equals_n(self):
        stream = list(range(10))
        result = self.rs(stream, k=10, seed=0)
        assert sorted(result) == stream

    def test_k_zero_raises(self):
        with pytest.raises(ValueError):
            self.rs([1, 2, 3], k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            self.rs([1, 2, 3], k=-1)

    def test_empty_stream_k_positive(self):
        # Empty stream: k > 0 but len(stream)=0 -> should return []
        result = self.rs([], k=5, seed=0)
        assert result == []

    def test_no_duplicates_for_unique_input(self):
        stream = list(range(200))
        result = self.rs(stream, k=50, seed=0)
        assert len(result) == len(set(result)), "Duplicate indices in result"

    def test_all_results_in_range(self):
        stream = list(range(100))
        result = self.rs(stream, k=30, seed=7)
        assert all(0 <= x < 100 for x in result)

    def test_deterministic_with_same_seed(self):
        s = list(range(100))
        r1 = self.rs(s, k=20, seed=99)
        r2 = self.rs(s, k=20, seed=99)
        assert r1 == r2

    def test_different_seeds_give_different_results(self):
        s = list(range(100))
        r1 = self.rs(s, k=20, seed=1)
        r2 = self.rs(s, k=20, seed=2)
        # Very unlikely to be identical for k=20 from 100 items
        assert r1 != r2

    def test_approximate_uniformity(self):
        """
        Each of 10 items should appear in the reservoir ~k/n fraction of runs.
        Uses a simulation over 500 seeds.
        """
        n, k = 10, 3
        stream = list(range(n))
        counts = {i: 0 for i in range(n)}
        for seed in range(500):
            for x in self.rs(stream, k=k, seed=seed):
                counts[x] += 1
        expected = 500 * k / n  # = 150
        for item, cnt in counts.items():
            assert abs(cnt - expected) < 60, (
                f"Item {item} count {cnt} deviates too much from {expected}"
            )

    def test_sample_client_data_fraction(self):
        indices = list(range(100))
        labels  = [i % 2 for i in indices]
        result  = self.scd(indices, labels, fraction=0.5, seed=42)
        assert len(result) == 50

    def test_sample_client_data_subset_of_input(self):
        indices = list(range(200))
        labels  = [0] * 200
        result  = self.scd(indices, labels, fraction=0.25, seed=0)
        assert set(result).issubset(set(indices))

    def test_sample_client_data_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            self.scd([1, 2, 3], [0, 0, 0], fraction=0.0)
        with pytest.raises(ValueError):
            self.scd([1, 2, 3], [0, 0, 0], fraction=1.5)

    def test_sample_client_data_fraction_1_returns_all(self):
        indices = list(range(50))
        labels  = [0] * 50
        result  = self.scd(indices, labels, fraction=1.0, seed=0)
        assert len(result) == 50


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataLoader:

    @requires_data
    def test_dataset_sizes(self):
        from data_loader import ChestXRayDataset
        train_ds = ChestXRayDataset(TRAIN_DIR, transform=None)
        test_ds  = ChestXRayDataset(TEST_DIR,  transform=None)
        assert len(train_ds) == 5216, f"Expected 5216, got {len(train_ds)}"
        assert len(test_ds)  == 624,  f"Expected 624, got {len(test_ds)}"

    @requires_data
    def test_class_label_mapping(self):
        from data_loader import ChestXRayDataset
        ds     = ChestXRayDataset(TRAIN_DIR, transform=None)
        labels = ds.get_labels()
        normal_count    = labels.count(0)
        pneumonia_count = labels.count(1)
        assert normal_count    == 1341
        assert pneumonia_count == 3875

    @requires_data
    def test_train_val_split_no_overlap(self):
        from data_loader import make_train_val_split
        train_idx, val_idx = make_train_val_split(TRAIN_DIR, 0.20, 42)
        assert set(train_idx).isdisjoint(set(val_idx)), "Train/val overlap!"

    @requires_data
    def test_train_val_split_full_coverage(self):
        from data_loader import ChestXRayDataset, make_train_val_split
        ds = ChestXRayDataset(TRAIN_DIR, transform=None)
        train_idx, val_idx = make_train_val_split(TRAIN_DIR, 0.20, 42)
        assert len(train_idx) + len(val_idx) == len(ds)

    @requires_data
    def test_train_val_split_reproducibility(self):
        from data_loader import make_train_val_split
        t1, v1 = make_train_val_split(TRAIN_DIR, 0.20, 42)
        t2, v2 = make_train_val_split(TRAIN_DIR, 0.20, 42)
        assert t1 == t2
        assert v1 == v2

    @requires_data
    def test_train_val_split_stratification(self):
        from data_loader import ChestXRayDataset, make_train_val_split
        ds     = ChestXRayDataset(TRAIN_DIR, transform=None)
        labels = ds.get_labels()
        train_idx, val_idx = make_train_val_split(TRAIN_DIR, 0.20, 42)

        def class_ratio(indices):
            n = sum(1 for i in indices if labels[i] == 1)
            return n / len(indices)

        tr_ratio  = class_ratio(train_idx)
        val_ratio = class_ratio(val_idx)
        overall   = class_ratio(list(range(len(labels))))
        # Both splits should be within 5pp of the overall ratio
        assert abs(tr_ratio  - overall) < 0.05
        assert abs(val_ratio - overall) < 0.05

    @requires_data
    def test_client_partitions_disjoint(self):
        from data_loader import ChestXRayDataset, make_train_val_split, partition_for_clients
        ds     = ChestXRayDataset(TRAIN_DIR, transform=None)
        labels = ds.get_labels()
        train_idx, _ = make_train_val_split(TRAIN_DIR, 0.20, 42)
        parts = partition_for_clients(train_idx, labels, n_clients=3, seed=42)
        all_idx = [i for p in parts for i in p]
        assert len(all_idx) == len(set(all_idx)), "Client partitions overlap!"

    @requires_data
    def test_client_partitions_full_coverage(self):
        from data_loader import ChestXRayDataset, make_train_val_split, partition_for_clients
        ds     = ChestXRayDataset(TRAIN_DIR, transform=None)
        labels = ds.get_labels()
        train_idx, _ = make_train_val_split(TRAIN_DIR, 0.20, 42)
        parts = partition_for_clients(train_idx, labels, n_clients=3, seed=42)
        all_idx = sorted([i for p in parts for i in p])
        assert all_idx == sorted(train_idx)

    @requires_data
    def test_reservoir_no_test_leakage(self):
        from data_loader import ChestXRayDataset, make_train_val_split, partition_for_clients
        from reservoir import sample_client_data
        ds     = ChestXRayDataset(TRAIN_DIR, transform=None)
        labels = ds.get_labels()
        train_idx, _ = make_train_val_split(TRAIN_DIR, 0.20, 42)
        parts = partition_for_clients(train_idx, labels, n_clients=3, seed=42)
        train_set = set(train_idx)
        for cid, part in enumerate(parts):
            sampled = sample_client_data(part, labels, fraction=0.5, seed=42 + cid)
            assert set(sampled).issubset(train_set), \
                f"Client {cid}: sampled indices include non-train indices"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class TestModel:

    def test_build_model_forward_pass(self):
        from model import build_model
        m = build_model(num_classes=2, freeze_backbone=True)
        dummy = torch.randn(2, 3, 224, 224)
        out = m(dummy)
        assert out.shape == (2, 2)

    def test_build_model_trainable_params(self):
        from model import build_model, count_parameters
        m = build_model(freeze_backbone=True)
        n = count_parameters(m)
        # Classifier head: 1280*2 + 2 = 2562
        assert n == 2562, f"Expected 2562 trainable params, got {n}"

    def test_build_model_opacus_compatible(self):
        from model import build_model
        from opacus.validators import ModuleValidator
        m = build_model()
        assert ModuleValidator.is_valid(m), "Model is not Opacus-compatible"

    def test_backbone_frozen(self):
        from model import build_model
        m = build_model(freeze_backbone=True)
        for name, param in m.features.named_parameters():
            assert not param.requires_grad, f"features.{name} should be frozen"

    def test_classifier_trainable(self):
        from model import build_model
        m = build_model(freeze_backbone=True)
        for name, param in m.classifier.named_parameters():
            assert param.requires_grad, f"classifier.{name} should be trainable"

    def test_state_dict_consistent_between_builds(self):
        from model import build_model
        m1 = build_model()
        m2 = build_model()
        assert set(m1.state_dict().keys()) == set(m2.state_dict().keys())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FEDAVG
# ═══════════════════════════════════════════════════════════════════════════════

class TestFedAvg:

    def setup_method(self):
        from federated import fedavg
        self.fedavg = fedavg

    def _tiny(self, w=0.0):
        m = nn.Linear(2, 2, bias=False)
        nn.init.constant_(m.weight, w)
        return m

    def test_equal_weights_averages_correctly(self):
        g = self._tiny(0.0)
        c1 = (self._tiny(2.0).state_dict(), 10)
        c2 = (self._tiny(0.0).state_dict(), 10)
        result = self.fedavg(g, [c1, c2])
        expected = torch.full((2, 2), 1.0)
        assert torch.allclose(result.weight, expected)

    def test_weighted_average(self):
        g  = self._tiny(0.0)
        c1 = (self._tiny(4.0).state_dict(), 30)
        c2 = (self._tiny(0.0).state_dict(), 10)
        result = self.fedavg(g, [c1, c2])
        expected = torch.full((2, 2), 3.0)
        assert torch.allclose(result.weight, expected)

    def test_empty_updates_raises(self):
        g = self._tiny()
        with pytest.raises(ValueError, match="no client updates"):
            self.fedavg(g, [])

    def test_zero_samples_raises(self):
        g  = self._tiny()
        sd = self._tiny().state_dict()
        with pytest.raises(ValueError):
            self.fedavg(g, [(sd, 0)])

    def test_mismatched_keys_raises(self):
        g  = self._tiny()
        # A model with an extra key not in the global model
        class BigNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc  = nn.Linear(2, 2, bias=False)
                self.fc2 = nn.Linear(2, 2, bias=False)
            def forward(self, x): return self.fc(x)
        with pytest.raises(ValueError):
            self.fedavg(g, [(BigNet().state_dict(), 10)])

    def test_result_loads_into_global_model(self):
        from model import build_model
        g  = build_model(freeze_backbone=False)
        sd = copy.deepcopy(g.state_dict())
        result = self.fedavg(g, [(sd, 100), (sd, 100)])
        # Should load without error
        g.load_state_dict(result.state_dict())

    def test_int_buffer_dtype_preserved(self):
        """num_batches_tracked is int64; result should still be int64."""
        # Use a model with a buffer (GroupNorm has weight/bias but no int buffer;
        # we test with a synthetic model containing an int buffer)
        class ModelWithIntBuf(nn.Module):
            def __init__(self, val):
                super().__init__()
                self.fc = nn.Linear(2, 2, bias=False)
                self.register_buffer("count", torch.tensor(val, dtype=torch.int64))
            def forward(self, x):
                return self.fc(x)

        g  = ModelWithIntBuf(0)
        c1 = (ModelWithIntBuf(10).state_dict(), 5)
        c2 = (ModelWithIntBuf(20).state_dict(), 5)
        res = self.fedavg(g, [c1, c2])
        assert res.count.dtype == torch.int64
        assert res.count.item() == 15  # (10+20)/2


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluate:

    def _metrics_from_known(self, y_true, y_pred, y_prob=None):
        """Run evaluate logic on known arrays without needing a real model."""
        from evaluate import evaluate_model
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Build a trivial model that outputs fixed predictions
        class FixedModel(nn.Module):
            def __init__(self, preds, probs):
                super().__init__()
                self.preds = torch.tensor(preds, dtype=torch.float32)
                self.probs = torch.tensor(probs, dtype=torch.float32)
            def forward(self, x):
                # Return logits: class0 score = 1-prob, class1 score = prob
                n = x.shape[0]
                return torch.stack([
                    1.0 - self.probs[:n],
                    self.probs[:n]
                ], dim=1)

        yt  = torch.tensor(y_true, dtype=torch.long)
        yp_  = y_prob if y_prob is not None else [float(p) for p in y_pred]
        ds  = TensorDataset(torch.zeros(len(y_true), 1), yt)
        loader = DataLoader(ds, batch_size=len(y_true))

        model = FixedModel(y_pred, yp_)
        device = torch.device("cpu")
        return evaluate_model(model, loader, device)

    def test_all_correct_binary(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.1, 0.9, 0.9]
        m = self._metrics_from_known(y_true, y_pred, y_prob)
        assert m["accuracy"]  == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["f1"]        == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["TN"] == 2
        assert m["TP"] == 2
        assert m["FP"] == 0
        assert m["FN"] == 0

    def test_all_pneumonia_predicted(self):
        """Model predicts PNEUMONIA for everything."""
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 1]
        y_prob = [0.9, 0.9, 0.9, 0.9]
        m = self._metrics_from_known(y_true, y_pred, y_prob)
        assert m["FP"] == 2
        assert m["TP"] == 2
        assert m["TN"] == 0
        assert m["FN"] == 0
        assert m["specificity"] == pytest.approx(0.0)
        assert m["recall"]      == pytest.approx(1.0)

    def test_confusion_matrix_shape_safe(self):
        """No crash when model predicts only one class."""
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        y_prob = [0.9, 0.9, 0.9, 0.9]
        # Should not raise; roc_auc should be nan
        m = self._metrics_from_known(y_true, y_pred, y_prob)
        import math
        assert math.isnan(m["roc_auc"]) or 0.0 <= m["roc_auc"] <= 1.0

    def test_positive_class_is_pneumonia(self):
        """Recall should count PNEUMONIA (label=1) detection."""
        y_true = [0, 1, 1, 1]     # 3 pneumonia cases
        y_pred = [0, 1, 1, 0]     # miss one pneumonia
        y_prob = [0.1, 0.9, 0.9, 0.3]
        m = self._metrics_from_known(y_true, y_pred, y_prob)
        assert m["FN"] == 1
        assert m["recall"] == pytest.approx(2 / 3)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RESULTS CSV
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassWeights:

    @requires_data
    def test_class_weights_computed_correctly(self):
        from data_loader import ChestXRayDataset, make_train_val_split, get_class_weights
        ds     = ChestXRayDataset(TRAIN_DIR, transform=None)
        labels = ds.get_labels()
        train_idx, _ = make_train_val_split(TRAIN_DIR, 0.20, 42)
        weights = get_class_weights(TRAIN_DIR, train_idx, labels)
        # NORMAL weight should be > 1 (minority), PNEUMONIA < 1 (majority)
        assert weights[0] > 1.0, "NORMAL weight should be > 1 (minority class)"
        assert weights[1] < 1.0, "PNEUMONIA weight should be < 1 (majority class)"
        # Weights should sum to 2 (standard inverse-frequency formula)
        n_norm  = sum(1 for i in train_idx if labels[i] == 0)
        n_pneu  = sum(1 for i in train_idx if labels[i] == 1)
        total   = n_norm + n_pneu
        assert abs(weights[0].item() - total / (2 * n_norm)) < 1e-4
        assert abs(weights[1].item() - total / (2 * n_pneu)) < 1e-4

    def test_class_weights_fix_majority_collapse(self):
        """
        Proves that weighted CE prevents collapse to majority-class prediction.
        Uses a tiny synthetic imbalanced dataset: 10 class-0, 40 class-1.
        """
        import torch, torch.nn as nn
        torch.manual_seed(0)
        n0, n1 = 10, 40
        X = torch.randn(n0 + n1, 4)
        y = torch.tensor([0] * n0 + [1] * n1)
        total = n0 + n1
        weights = torch.tensor([total / (2 * n0), total / (2 * n1)])

        def train_and_predict(use_weights):
            model = nn.Linear(4, 2)
            opt   = torch.optim.Adam(model.parameters(), lr=0.1)
            crit  = (nn.CrossEntropyLoss(weight=weights)
                     if use_weights else nn.CrossEntropyLoss())
            for _ in range(30):
                opt.zero_grad()
                crit(model(X), y).backward()
                opt.step()
            with torch.no_grad():
                preds = model(X).argmax(1)
            return (preds == 0).sum().item()

        n_normal_plain    = train_and_predict(use_weights=False)
        n_normal_weighted = train_and_predict(use_weights=True)
        # Without weights, model likely collapses to all-class-1 (0 NORMAL predictions)
        # With weights, model should predict at least some NORMAL
        assert n_normal_weighted > n_normal_plain, (
            f"Weighted CE should predict more NORMAL samples "
            f"(plain={n_normal_plain}, weighted={n_normal_weighted})"
        )

    def test_save_result_creates_file(self):
        from evaluate import save_result
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "metrics.csv")
            row = {"experiment": "test", "accuracy": 0.9, "f1": 0.88}
            save_result(row, csv_path)
            assert os.path.exists(csv_path)

    def test_save_result_deduplicates(self):
        import pandas as pd
        from evaluate import save_result
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "metrics.csv")
            row1 = {"experiment": "E1", "accuracy": 0.80}
            row2 = {"experiment": "E1", "accuracy": 0.90}
            save_result(row1, csv_path)
            save_result(row2, csv_path)
            df = pd.read_csv(csv_path)
            e1_rows = df[df["experiment"] == "E1"]
            assert len(e1_rows) == 1, "Should overwrite, not append duplicate"
            assert e1_rows["accuracy"].iloc[0] == pytest.approx(0.90)

    def test_save_result_preserves_other_rows(self):
        import pandas as pd
        from evaluate import save_result
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "metrics.csv")
            save_result({"experiment": "E1", "accuracy": 0.80}, csv_path)
            save_result({"experiment": "E2", "accuracy": 0.75}, csv_path)
            save_result({"experiment": "E1", "accuracy": 0.90}, csv_path)
            df = pd.read_csv(csv_path)
            assert len(df) == 2
            assert set(df["experiment"]) == {"E1", "E2"}


# ═══════════════════════════════════════════════════════════════════════════════
# 8. DP TRAINING (small synthetic model)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDPTraining:
    """
    Tests DP-SGD mechanics using a tiny synthetic model.
    Does NOT use the real dataset or MobileNetV2.
    """

    def _make_dp_objects(self, dataset_size=20, batch_size=4):
        from opacus import PrivacyEngine
        model = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))
        engine = PrivacyEngine(accountant="rdp")
        opt    = torch.optim.Adam(model.parameters(), lr=0.01)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(dataset_size, 8),
                torch.randint(0, 2, (dataset_size,)),
            ),
            batch_size=batch_size,
        )
        dp_model, dp_opt, dp_loader = engine.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=True,
        )
        return engine, dp_model, dp_opt, dp_loader

    def test_make_private_module_attribute(self):
        engine, dp_model, _, _ = self._make_dp_objects()
        assert hasattr(dp_model, "_module"), (
            "_module not found on GradSampleModule — Opacus API may have changed"
        )

    def test_module_state_dict_keys_match_plain(self):
        from opacus import PrivacyEngine
        plain = nn.Linear(4, 2)
        engine = PrivacyEngine(accountant="rdp")
        opt    = torch.optim.Adam(plain.parameters(), lr=0.01)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(8, 4), torch.randint(0, 2, (8,))
            ), batch_size=4
        )
        dp_model, _, _ = engine.make_private(
            module=plain, optimizer=opt, data_loader=loader,
            noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=True,
        )
        assert list(dp_model._module.state_dict().keys()) == list(plain.state_dict().keys())

    def test_accountant_accumulates_across_steps(self):
        engine, dp_model, dp_opt, dp_loader = self._make_dp_objects()
        crit = nn.CrossEntropyLoss()
        step_count = 0
        for x, y in dp_loader:
            dp_opt.zero_grad()
            crit(dp_model(x), y).backward()
            dp_opt.step()
            step_count += 1
            if step_count >= 3:
                break
        total_steps = sum(s for _, _, s in engine.accountant.history)
        assert total_steps >= 1, "Accountant recorded no steps"

    def test_epsilon_nondecreasing_across_rounds(self):
        engine, dp_model, dp_opt, dp_loader = self._make_dp_objects()
        crit     = nn.CrossEntropyLoss()
        epsilons = []

        for rnd in range(2):
            for x, y in dp_loader:
                dp_opt.zero_grad()
                crit(dp_model(x), y).backward()
                dp_opt.step()
            epsilons.append(engine.get_epsilon(delta=1e-5))

        assert epsilons[1] >= epsilons[0], (
            f"Epsilon decreased: {epsilons[0]:.4f} -> {epsilons[1]:.4f}. "
            "Accountant may have reset."
        )

    def test_epsilon_is_finite_and_nonnegative(self):
        engine, dp_model, dp_opt, dp_loader = self._make_dp_objects()
        crit = nn.CrossEntropyLoss()
        for x, y in dp_loader:
            dp_opt.zero_grad()
            crit(dp_model(x), y).backward()
            dp_opt.step()
            break
        eps = engine.get_epsilon(delta=1e-5)
        import math
        assert math.isfinite(eps) and eps >= 0

    def test_weight_sync_via_module_load_state_dict(self):
        """
        Loading global weights into dp_model._module between rounds
        should work correctly and not affect accountant history.
        """
        engine, dp_model, dp_opt, dp_loader = self._make_dp_objects()
        crit  = nn.CrossEntropyLoss()

        # Round 1
        for x, y in dp_loader:
            dp_opt.zero_grad()
            crit(dp_model(x), y).backward()
            dp_opt.step()
            break

        history_after_r1 = list(engine.accountant.history)

        # Load new global weights (simulate FedAvg result)
        new_weights = copy.deepcopy(dp_model._module.state_dict())
        for k in new_weights:
            new_weights[k] = torch.zeros_like(new_weights[k])
        dp_model._module.load_state_dict(new_weights)

        # Accountant should not have changed
        assert engine.accountant.history == history_after_r1, (
            "Accountant history changed after weight reload — unexpected reset"
        )

        # Round 2 — should still accumulate
        for x, y in dp_loader:
            dp_opt.zero_grad()
            crit(dp_model(x), y).backward()
            dp_opt.step()
            break

        eps = engine.get_epsilon(delta=1e-5)
        import math
        assert math.isfinite(eps) and eps >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SMOKE TEST — end-to-end tiny run (requires dataset)
# ═══════════════════════════════════════════════════════════════════════════════

@requires_data
@pytest.mark.slow
def test_smoke_e1_runs_one_epoch():
    """E1 runs for 1 epoch without error and saves a result row."""
    import pandas as pd
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "metrics.csv")
        # Temporarily override METRICS_CSV via monkey-patching
        import evaluate
        import train_centralized
        original_csv = evaluate.RESULTS_DIR

        # Patch just the CSV path
        import config as cfg
        old_csv = cfg.METRICS_CSV
        cfg.METRICS_CSV = csv_path
        train_centralized.METRICS_CSV = csv_path
        train_centralized.EPOCHS = 1

        try:
            metrics = train_centralized.run()
            assert "accuracy" in metrics
            assert 0.0 <= metrics["accuracy"] <= 1.0
            df = pd.read_csv(csv_path)
            assert len(df) == 1
            assert df["experiment"].iloc[0] == "E1 - Centralized"
        finally:
            cfg.METRICS_CSV = old_csv
            train_centralized.METRICS_CSV = old_csv
            train_centralized.EPOCHS = 5

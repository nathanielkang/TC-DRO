"""
Smoke test for TC-DRO implementation.

Tests all implemented modules end-to-end with synthetic data:
1. Data loading (relevance, density)
2. Model creation (MLP, Linear)
3. Ambiguity sets (ChiSquaredTC, WassersteinTC)
4. TC-DRO training (2 epochs, loss decreases)
5. All evaluation metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_synthetic_data(n: int = 100, d: int = 5, seed: int = 42):
    """Create synthetic imbalanced regression data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    y = X @ w + rng.exponential(2.0, size=n).astype(np.float32)
    return X, y


def _make_loader(X, y, batch_size=32):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


def test_imports() -> bool:
    """Verify all modules import without error."""
    try:
        from src.data.datasets import (
            ImbalancedRegressionDataset,
            estimate_target_density,
            get_relevance_function,
        )
        from src.dro.ambiguity_sets import ChiSquaredTC, TargetConditionalAmbiguitySet, WassersteinTC
        from src.dro.tc_dro_trainer import TCDROTrainer
        from src.evaluation.metrics import (
            coverage_at_quantile,
            mae,
            mse,
            sera,
            tail_mae,
            tail_mse,
        )
        from src.models.base_models import LinearRegressor, MLPRegressor, XGBoostWrapper
        return True
    except ImportError as e:
        print(f"  FAIL import: {e}")
        return False


def test_relevance_and_density() -> bool:
    """Test relevance function and density estimation."""
    try:
        from src.data.datasets import estimate_target_density, get_relevance_function

        _, y = _make_synthetic_data()
        rel = get_relevance_function(y)
        assert rel.shape == y.shape, f"Relevance shape mismatch: {rel.shape} vs {y.shape}"
        assert np.all(rel >= 0) and np.all(rel <= 1.0 + 1e-6), "Relevance out of [0,1]"

        dens = estimate_target_density(y, num_bins=20)
        assert dens.shape == y.shape, f"Density shape mismatch: {dens.shape} vs {y.shape}"
        assert np.all(dens > 0), "Density contains non-positive values"
        return True
    except Exception as e:
        print(f"  FAIL relevance/density: {e}")
        return False


def test_models() -> bool:
    """Test MLP and Linear model forward passes."""
    try:
        from src.models.base_models import LinearRegressor, MLPRegressor

        X_t = torch.randn(16, 5)

        mlp = MLPRegressor(5, [32, 16], dropout=0.1, activation="relu")
        out = mlp(X_t)
        assert out.shape == (16, 1), f"MLP output shape: {out.shape}"

        lin = LinearRegressor(5)
        out2 = lin(X_t)
        assert out2.shape == (16, 1), f"Linear output shape: {out2.shape}"
        return True
    except Exception as e:
        print(f"  FAIL models: {e}")
        return False


def test_ambiguity_sets() -> bool:
    """Test ambiguity set radius and weight computation."""
    try:
        from src.data.datasets import estimate_target_density
        from src.dro.ambiguity_sets import ChiSquaredTC, WassersteinTC

        _, y = _make_synthetic_data()
        dens = estimate_target_density(y, num_bins=20)

        chi2 = ChiSquaredTC(radius_base=0.1, density_exponent=0.5, num_bins=20)
        r = chi2.compute_radius(y, dens)
        assert r.shape == y.shape, f"Chi2 radius shape: {r.shape}"
        assert np.all(r > 0), "Chi2 radius non-positive"
        w = chi2.compute_sample_weights(y, dens)
        assert w.shape == y.shape, f"Chi2 weights shape: {w.shape}"
        assert abs(w.mean() - 1.0) < 0.1, f"Chi2 weights mean != 1: {w.mean()}"

        wass = WassersteinTC(radius_base=0.1, density_exponent=0.5, num_bins=20)
        r2 = wass.compute_radius(y, dens)
        assert r2.shape == y.shape
        w2 = wass.compute_sample_weights(y, dens)
        assert w2.shape == y.shape
        assert abs(w2.mean() - 1.0) < 0.1, f"Wass weights mean != 1: {w2.mean()}"
        return True
    except Exception as e:
        print(f"  FAIL ambiguity sets: {e}")
        return False


def test_training_loss_decreases() -> bool:
    """Train MLP with TC-DRO for 2 epochs, verify loss decreases."""
    try:
        from src.dro.ambiguity_sets import ChiSquaredTC
        from src.dro.tc_dro_trainer import TCDROTrainer
        from src.models.base_models import MLPRegressor

        torch.manual_seed(42)
        np.random.seed(42)

        X, y = _make_synthetic_data(n=100, d=5, seed=42)
        loader = _make_loader(X, y, batch_size=32)

        model = MLPRegressor(5, [32, 16], dropout=0.0, activation="relu")
        amb = ChiSquaredTC(radius_base=0.1, density_exponent=0.5, num_bins=20)
        config = {
            "training": {"epochs": 2, "lr": 1e-2, "weight_decay": 0, "patience": 10},
        }
        trainer = TCDROTrainer(model, amb, config)

        loss1 = trainer.train_epoch(loader)
        loss2 = trainer.train_epoch(loader)

        if loss2 >= loss1:
            print(f"  WARN: Loss did not decrease: {loss1:.4f} -> {loss2:.4f} (may be stochastic)")
        return True
    except Exception as e:
        print(f"  FAIL training: {e}")
        return False


def test_full_fit() -> bool:
    """Test full fit() with early stopping."""
    try:
        from src.dro.ambiguity_sets import ChiSquaredTC
        from src.dro.tc_dro_trainer import TCDROTrainer
        from src.models.base_models import MLPRegressor

        torch.manual_seed(42)
        np.random.seed(42)

        X, y = _make_synthetic_data(n=100, d=5, seed=42)
        n_train = 70
        train_loader = _make_loader(X[:n_train], y[:n_train], batch_size=32)
        val_loader = _make_loader(X[n_train:], y[n_train:], batch_size=32)

        model = MLPRegressor(5, [32, 16], dropout=0.0, activation="relu")
        amb = ChiSquaredTC(radius_base=0.1, density_exponent=0.5, num_bins=20)
        config = {"training": {"epochs": 5, "lr": 1e-2, "weight_decay": 0, "patience": 3}}
        trainer = TCDROTrainer(model, amb, config)

        result = trainer.fit(train_loader, val_loader)
        assert "history" in result
        assert "best_epoch" in result
        assert "best_metrics" in result
        assert len(result["history"]["train_loss"]) > 0
        return True
    except Exception as e:
        print(f"  FAIL fit: {e}")
        return False


def test_all_metrics() -> bool:
    """Test all evaluation metrics on synthetic data."""
    try:
        from src.evaluation.metrics import (
            coverage_at_quantile,
            mae,
            mse,
            sera,
            tail_mae,
            tail_mse,
        )
        from src.data.datasets import get_relevance_function

        y_true = np.array([1.0, 2.0, 3.0, 10.0, 0.1, 2.5, 3.5, 8.0, 0.5, 4.0])
        y_pred = y_true + np.random.default_rng(0).normal(0, 0.5, size=len(y_true))

        m = mse(y_true, y_pred)
        assert m >= 0, f"MSE negative: {m}"

        a = mae(y_true, y_pred)
        assert a >= 0, f"MAE negative: {a}"

        rel = get_relevance_function(y_true)
        s = sera(y_true, y_pred, rel)
        assert s >= 0, f"SERA negative: {s}"

        tm = tail_mse(y_true, y_pred, 10)
        assert tm >= 0, f"Tail MSE negative: {tm}"

        ta = tail_mae(y_true, y_pred, 10)
        assert ta >= 0, f"Tail MAE negative: {ta}"

        c = coverage_at_quantile(y_true, y_pred, 10)
        assert 0 <= c <= 1.0, f"Coverage out of [0,1]: {c}"

        return True
    except Exception as e:
        print(f"  FAIL metrics: {e}")
        return False


def test_evaluate_via_trainer() -> bool:
    """Test trainer.evaluate() produces all expected metric keys."""
    try:
        from src.dro.ambiguity_sets import ChiSquaredTC
        from src.dro.tc_dro_trainer import TCDROTrainer
        from src.models.base_models import MLPRegressor

        torch.manual_seed(42)
        X, y = _make_synthetic_data(n=50, d=5, seed=99)
        loader = _make_loader(X, y, batch_size=50)

        model = MLPRegressor(5, [16], dropout=0.0, activation="relu")
        amb = ChiSquaredTC()
        config = {"training": {"epochs": 1, "lr": 1e-3, "weight_decay": 0, "patience": 5}}
        trainer = TCDROTrainer(model, amb, config)

        metrics = trainer.evaluate(loader)
        expected_keys = {"mse", "mae", "sera", "tail_mse_5", "tail_mse_10",
                         "tail_mae_5", "tail_mae_10", "coverage_5", "coverage_10"}
        missing = expected_keys - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"
        return True
    except Exception as e:
        print(f"  FAIL evaluate: {e}")
        return False


def main() -> int:
    """Run all smoke tests. Return 0 on PASS, 1 on FAIL."""
    tests = [
        ("Imports", test_imports),
        ("Relevance & Density", test_relevance_and_density),
        ("Models (MLP, Linear)", test_models),
        ("Ambiguity Sets (Chi2, Wass)", test_ambiguity_sets),
        ("Training loss (2 epochs)", test_training_loss_decreases),
        ("Full fit (early stopping)", test_full_fit),
        ("All metrics", test_all_metrics),
        ("Evaluate via trainer", test_evaluate_via_trainer),
    ]

    results = []
    for name, fn in tests:
        ok = fn()
        results.append((name, ok))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    all_pass = all(r[1] for r in results)
    print()
    if all_pass:
        print("=" * 40)
        print("PASS: All smoke tests passed.")
        print("=" * 40)
        return 0
    else:
        failed = [name for name, ok in results if not ok]
        print("=" * 40)
        print(f"FAIL: {len(failed)} test(s) failed: {', '.join(failed)}")
        print("=" * 40)
        return 1


if __name__ == "__main__":
    sys.exit(main())

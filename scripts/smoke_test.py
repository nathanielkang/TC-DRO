"""Quick smoke test: 2 epochs on tiny synthetic data for all methods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dro.ambiguity_sets import (
    ChiSquaredStandard,
    ChiSquaredTC,
    ERMAmbiguity,
    WassersteinStandard,
    WassersteinTC,
)
from src.dro.tc_dro_trainer import TCDROTrainer
from src.models.base_models import MLPRegressor


def _make_loader(n: int = 200, d: int = 5, batch_size: int = 32):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] + rng.exponential(1.0, n)).astype(np.float32)
    ds = torch.utils.data.TensorDataset(
        torch.tensor(X), torch.tensor(y),
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True), y


def test_method(name: str, ambiguity, y_train: np.ndarray) -> None:
    model = MLPRegressor(input_dim=5, hidden_dims=[32, 32])
    config = {
        "training": {"epochs": 2, "lr": 1e-3, "weight_decay": 0, "patience": 5},
        "dro": {"loss_clip": 9.0},
    }
    trainer = TCDROTrainer(model, ambiguity, config)
    trainer.set_y_train(y_train)

    loader, _ = _make_loader()
    result = trainer.fit(loader, loader)
    metrics = trainer.evaluate(loader)
    print(
        f"  {name}: epoch={result['best_epoch']}, mse={metrics['mse']:.4f}, "
        f"sera={metrics['sera']:.4f}, M={trainer.loss_clip:.1f}"
    )


def test_baseline(name: str, trainer_cls, y_train: np.ndarray) -> None:
    from src.models.base_models import MLPRegressor as MLP

    model = MLP(input_dim=5, hidden_dims=[32, 32])
    config = {
        "training": {"epochs": 2, "lr": 1e-3, "weight_decay": 0, "patience": 5},
        "lds_fds": {"sigma": 2.0, "kernel_size": 5, "fds_feature_dim": 32, "fds_bins": 20},
        "dense_loss": {"bw_method": "silverman"},
        "conr": {"temperature": 0.07, "conr_weight": 0.1, "threshold": 1.0},
    }
    trainer = trainer_cls(model, config)
    trainer.set_y_train(y_train)
    loader, _ = _make_loader()
    result = trainer.fit(loader, loader)
    metrics = trainer.evaluate(loader)
    print(f"  {name}: epoch={result['best_epoch']}, mse={metrics['mse']:.4f}, sera={metrics['sera']:.4f}")


def test_loss_clip(M: float = 9.0) -> None:
    """Truncation must bound the training objective (ERM weights are all 1)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 5)).astype(np.float32)
    y = np.full(64, 100.0, dtype=np.float32)
    ds = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = torch.utils.data.DataLoader(ds, batch_size=32)

    model = MLPRegressor(input_dim=5, hidden_dims=[16, 16])
    config = {
        "training": {"epochs": 1, "lr": 1e-3, "weight_decay": 0, "patience": 1},
        "dro": {"loss_clip": M},
    }
    trainer = TCDROTrainer(model, ERMAmbiguity(), config)
    trainer.set_y_train(y)
    train_loss = trainer.train_epoch(loader)
    assert train_loss <= M + 1e-6, f"truncation inactive: train_loss={train_loss}"
    print(f"  bounded objective: train_loss={train_loss:.4f} <= M={M:.1f}")


def main() -> None:
    print("=== TC-DRO Smoke Test ===\n")
    _, y_train = _make_loader()

    print("Loss truncation:")
    test_loss_clip()
    print()

    print("DRO methods:")
    test_method("ERM", ERMAmbiguity(), y_train)
    test_method("chi2-DRO (standard)", ChiSquaredStandard(radius=0.1), y_train)
    test_method("W-DRO (standard)", WassersteinStandard(radius=0.1), y_train)
    test_method("TC-DRO-chi2", ChiSquaredTC(radius_base=0.1, density_exponent=0.5, num_bins=5), y_train)
    test_method("TC-DRO-Wass", WassersteinTC(radius_base=0.1, density_exponent=0.5, num_bins=5), y_train)

    print("\nBaseline methods:")
    from src.baselines.lds_fds import LDSFDSTrainer
    from src.baselines.dense_loss import DenseLossTrainer
    from src.baselines.conr import ConRTrainer

    test_baseline("LDS+FDS", LDSFDSTrainer, y_train)
    test_baseline("DenseLoss", DenseLossTrainer, y_train)
    test_baseline("ConR", ConRTrainer, y_train)

    print("\n=== All smoke tests PASSED ===")


if __name__ == "__main__":
    main()

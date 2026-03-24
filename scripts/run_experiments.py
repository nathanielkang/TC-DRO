"""Experiment runner for TC-DRO imbalanced regression."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import ImbalancedRegressionDataset, estimate_target_density
from src.dro.ambiguity_sets import ChiSquaredTC, WassersteinTC
from src.dro.tc_dro_trainer import TCDROTrainer
from src.models.base_models import LinearRegressor, MLPRegressor, XGBoostWrapper

ALL_DATASETS = [
    "abalone", "ailerons", "accel", "bank8FM",
    "cpu_act", "delta_ailerons", "machine_cpu", "airfoil",
]

ALL_METHODS = ["tc-dro-chi2", "tc-dro-wass", "erm", "dro-chi2", "dro-wass"]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed args (config path, output dir, seeds, datasets, methods).
    """
    parser = argparse.ArgumentParser(description="TC-DRO Experiment Runner")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--datasets", nargs="+", type=str, default=ALL_DATASETS)
    parser.add_argument("--methods", nargs="+", type=str, default=ALL_METHODS)
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load YAML configuration.

    Args:
        config_path: Path to config YAML.

    Returns:
        Config dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """Create a simple torch DataLoader from numpy arrays."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _create_ambiguity_set(method: str, dro_cfg: dict) -> Any:
    """Create the appropriate ambiguity set for the method.

    TC methods use density_exponent from config (target-conditional weighting).
    Standard DRO methods use density_exponent=0.0 (uniform weights).
    ERM uses radius=0 and density_exponent=0 (no robustness, no reweighting).
    """
    r = dro_cfg.get("radius_base", 0.1)
    e = dro_cfg.get("density_exponent", 0.5)
    b = dro_cfg.get("num_bins", 50)

    if method == "tc-dro-chi2":
        return ChiSquaredTC(radius_base=r, density_exponent=e, num_bins=b)
    elif method == "dro-chi2":
        return ChiSquaredTC(radius_base=r, density_exponent=0.0, num_bins=b)
    elif method == "tc-dro-wass":
        return WassersteinTC(radius_base=r, density_exponent=e, num_bins=b)
    elif method == "dro-wass":
        return WassersteinTC(radius_base=r, density_exponent=0.0, num_bins=b)
    elif method == "erm":
        return ChiSquaredTC(radius_base=0.0, density_exponent=0.0, num_bins=b)
    else:
        return ChiSquaredTC(radius_base=r, density_exponent=e, num_bins=b)


def _create_model(model_cfg: dict, input_dim: int) -> Any:
    """Create model from config."""
    arch = model_cfg.get("architecture", "mlp")
    if arch == "mlp":
        return MLPRegressor(
            input_dim=input_dim,
            hidden_dims=model_cfg.get("hidden_dims", [64, 32]),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "relu"),
        )
    elif arch == "xgboost":
        return XGBoostWrapper(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
    elif arch == "linear":
        return LinearRegressor(input_dim=input_dim)
    else:
        return MLPRegressor(input_dim=input_dim, hidden_dims=[64, 32])


def run_single_experiment(
    config: dict[str, Any],
    dataset_name: str,
    method: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Run one experiment: load data, create model, train, evaluate.

    Args:
        config: Full config dict.
        dataset_name: Dataset to use.
        method: DRO method name.
        seed: Random seed.
        output_dir: Directory for results.

    Returns:
        Results dict (metrics, config, seed, dataset).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds = ImbalancedRegressionDataset(
        name=dataset_name,
        test_ratio=config["dataset"].get("test_ratio", 0.2),
        val_ratio=config["dataset"].get("val_ratio", 0.1),
        random_seed=seed,
    )
    splits = ds.load()

    input_dim = splits["X_train"].shape[1]
    batch_size = config["training"].get("batch_size", 64)

    train_loader = _make_dataloader(splits["X_train"], splits["y_train"], batch_size, shuffle=True)
    val_loader = _make_dataloader(splits["X_val"], splits["y_val"], batch_size, shuffle=False)
    test_loader = _make_dataloader(splits["X_test"], splits["y_test"], batch_size, shuffle=False)

    model = _create_model(config["model"], input_dim)
    ambiguity = _create_ambiguity_set(method, config["dro"])

    trainer = TCDROTrainer(model, ambiguity, config)

    t0 = time.time()
    fit_result = trainer.fit(train_loader, val_loader)
    elapsed = time.time() - t0

    test_metrics = trainer.evaluate(test_loader)

    result = {
        "dataset": dataset_name,
        "method": method,
        "seed": seed,
        "best_epoch": fit_result["best_epoch"],
        "train_time_s": round(elapsed, 2),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    out_path = output_dir / f"{dataset_name}_{method}_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    """
    Main entry: parse config, run experiments across seeds/datasets/methods,
    save results to JSON/CSV.
    """
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total = len(args.seeds) * len(args.datasets) * len(args.methods)
    count = 0

    for seed in args.seeds:
        for ds_name in args.datasets:
            for method in args.methods:
                count += 1
                print(f"[{count}/{total}] seed={seed} dataset={ds_name} method={method}")
                try:
                    result = run_single_experiment(config, ds_name, method, seed, output_dir)
                    all_results.append(result)
                    print(f"  -> test_mse={result.get('test_mse', 'N/A'):.4f}")
                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    all_results.append({
                        "dataset": ds_name, "method": method, "seed": seed, "error": str(e),
                    })

    df = pd.DataFrame(all_results)
    csv_path = output_dir / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

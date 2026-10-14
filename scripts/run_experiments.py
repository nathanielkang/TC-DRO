"""Experiment runner for TC-DRO imbalanced regression.

Runs the configured method list × datasets × seeds; writes per-run JSON and a
summary CSV. Baseline hyperparameters are searched via validation SERA where
grid sections exist in the YAML config.
"""

from __future__ import annotations

import argparse
import itertools
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

from src.data.datasets import ImbalancedRegressionDataset
from src.dro.ambiguity_sets import (
    ChiSquaredStandard,
    ChiSquaredTC,
    ERMAmbiguity,
    WassersteinStandard,
    WassersteinTC,
)
from src.dro.tc_dro_trainer import TCDROTrainer
from src.models.base_models import MLPRegressor

ALL_DATASETS = [
    "abalone", "ailerons", "accel", "bank8FM",
    "cpu_act", "delta_ailerons", "machine_cpu",
]

ALL_METHODS = [
    "erm", "dro-chi2", "dro-wass",
    "lds-fds", "dense-loss", "conr",
    "tc-dro-chi2", "tc-dro-wass",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TC-DRO Experiment Runner")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    parser.add_argument("--datasets", nargs="+", type=str, default=ALL_DATASETS)
    parser.add_argument("--methods", nargs="+", type=str, default=ALL_METHODS)
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_dataloader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True,
):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _create_model(model_cfg: dict, input_dim: int) -> MLPRegressor:
    return MLPRegressor(
        input_dim=input_dim,
        hidden_dims=model_cfg.get("hidden_dims", [256, 256, 256, 256]),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "relu"),
    )


def _run_baseline_with_grid(
    method: str,
    config: dict,
    splits: dict,
    input_dim: int,
    batch_size: int,
) -> dict[str, Any]:
    """Run a baseline method with hyperparameter grid search on val SERA."""
    train_loader = _make_dataloader(splits["X_train"], splits["y_train"], batch_size)
    val_loader = _make_dataloader(splits["X_val"], splits["y_val"], batch_size, shuffle=False)
    test_loader = _make_dataloader(splits["X_test"], splits["y_test"], batch_size, shuffle=False)
    grids = config.get("baseline_grids", {})

    if method == "lds-fds":
        from src.baselines.lds_fds import LDSFDSTrainer

        grid_cfg = grids.get("lds_fds", {})
        sigmas = grid_cfg.get("sigma", [2.0])
        ks_sizes = grid_cfg.get("kernel_size", [5])

        best_sera = float("inf")
        best_trainer = None
        for sigma, ks in itertools.product(sigmas, ks_sizes):
            model = _create_model(config["model"], input_dim)
            trial_config = {**config, "lds_fds": {"sigma": sigma, "kernel_size": ks,
                            "fds_feature_dim": grid_cfg.get("fds_feature_dim", 256),
                            "fds_bins": grid_cfg.get("fds_bins", 100)}}
            trainer = LDSFDSTrainer(model, trial_config)
            trainer.set_y_train(splits["y_train"])
            trainer.fit(train_loader, val_loader)
            val_metrics = trainer.evaluate(val_loader)
            if val_metrics["sera"] < best_sera:
                best_sera = val_metrics["sera"]
                best_trainer = trainer

        test_metrics = best_trainer.evaluate(test_loader)
        return test_metrics

    elif method == "dense-loss":
        from src.baselines.dense_loss import DenseLossTrainer

        grid_cfg = grids.get("dense_loss", {})
        bws = grid_cfg.get("bandwidth", ["silverman", "scott"])

        best_sera = float("inf")
        best_trainer = None
        for bw in bws:
            bw_str = str(bw).lower()
            model = _create_model(config["model"], input_dim)
            trial_config = {**config, "dense_loss": {"bw_method": bw_str}}
            trainer = DenseLossTrainer(model, trial_config)
            trainer.set_y_train(splits["y_train"])
            trainer.fit(train_loader, val_loader)
            val_metrics = trainer.evaluate(val_loader)
            if val_metrics["sera"] < best_sera:
                best_sera = val_metrics["sera"]
                best_trainer = trainer

        return best_trainer.evaluate(test_loader)

    elif method == "conr":
        from src.baselines.conr import ConRTrainer

        grid_cfg = grids.get("conr", {})
        temps = grid_cfg.get("temperature", [0.07])
        weights = grid_cfg.get("conr_weight", [0.1])
        thresholds = grid_cfg.get("threshold", [1.0])

        best_sera = float("inf")
        best_trainer = None
        for temp, cw, thr in itertools.product(temps, weights, thresholds):
            model = _create_model(config["model"], input_dim)
            trial_config = {**config, "conr": {"temperature": temp,
                            "conr_weight": cw, "threshold": thr}}
            trainer = ConRTrainer(model, trial_config)
            trainer.set_y_train(splits["y_train"])
            trainer.fit(train_loader, val_loader)
            val_metrics = trainer.evaluate(val_loader)
            if val_metrics["sera"] < best_sera:
                best_sera = val_metrics["sera"]
                best_trainer = trainer

        test_metrics = best_trainer.evaluate(test_loader)
        return test_metrics

    raise ValueError(f"Unknown baseline method: {method}")


def run_single_experiment(
    config: dict[str, Any],
    dataset_name: str,
    method: str,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one experiment: load data, create model/ambiguity, train, evaluate."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds = ImbalancedRegressionDataset(
        name=dataset_name,
        test_ratio=config["dataset"].get("test_ratio", 0.15),
        val_ratio=config["dataset"].get("val_ratio", 0.15),
        random_seed=seed,
    )
    splits = ds.load()
    input_dim = splits["X_train"].shape[1]
    batch_size = config["training"].get("batch_size", 64)

    t0 = time.time()

    if method in ("lds-fds", "dense-loss", "conr"):
        test_metrics = _run_baseline_with_grid(
            method, config, splits, input_dim, batch_size,
        )
        elapsed = time.time() - t0
        result = {
            "dataset": dataset_name,
            "method": method,
            "seed": seed,
            "best_epoch": -1,
            "train_time_s": round(elapsed, 2),
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
    else:
        train_loader = _make_dataloader(
            splits["X_train"], splits["y_train"], batch_size, shuffle=True,
        )
        val_loader = _make_dataloader(
            splits["X_val"], splits["y_val"], batch_size, shuffle=False,
        )
        test_loader = _make_dataloader(
            splits["X_test"], splits["y_test"], batch_size, shuffle=False,
        )

        model = _create_model(config["model"], input_dim)
        dro_cfg = config["dro"]

        if method == "tc-dro-chi2":
            ambiguity = ChiSquaredTC(
                radius_base=dro_cfg["radius_base"],
                density_exponent=dro_cfg["density_exponent"],
                num_bins=dro_cfg["num_bins"],
                weight_clip=dro_cfg.get("weight_clip", 10.0),
            )
        elif method == "tc-dro-wass":
            ambiguity = WassersteinTC(
                radius_base=dro_cfg["radius_base"],
                density_exponent=dro_cfg["density_exponent"],
                num_bins=dro_cfg["num_bins"],
                weight_clip=dro_cfg.get("weight_clip", 10.0),
            )
        elif method == "dro-chi2":
            ambiguity = ChiSquaredStandard(
                radius=dro_cfg["radius_base"],
                weight_clip=dro_cfg.get("weight_clip", 10.0),
            )
        elif method == "dro-wass":
            ambiguity = WassersteinStandard(
                radius=dro_cfg["radius_base"],
                weight_clip=dro_cfg.get("weight_clip", 10.0),
            )
        elif method == "erm":
            ambiguity = ERMAmbiguity()
        else:
            raise ValueError(f"Unknown method: {method}")

        trainer = TCDROTrainer(model, ambiguity, config)
        trainer.set_y_train(splits["y_train"])

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


def summarize_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Compute mean ± std per method per dataset and save summary."""
    metric_cols = [c for c in df.columns if c.startswith("test_")]
    if not metric_cols:
        return

    summary_rows = []
    for (dataset, method), grp in df.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method, "n_seeds": len(grp)}
        for mc in metric_cols:
            vals = grp[mc].dropna()
            row[f"{mc}_mean"] = vals.mean()
            row[f"{mc}_std"] = vals.std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary_mean_std.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary (mean ± std) saved to {summary_path}")


def main() -> None:
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
                    result = run_single_experiment(
                        config, ds_name, method, seed, output_dir,
                    )
                    all_results.append(result)
                    print(f"  -> test_mse={result.get('test_mse', 'N/A'):.4f}")
                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    all_results.append({
                        "dataset": ds_name, "method": method,
                        "seed": seed, "error": str(e),
                    })

    df = pd.DataFrame(all_results)
    csv_path = output_dir / "experiment_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

    summarize_results(df, output_dir)


if __name__ == "__main__":
    main()

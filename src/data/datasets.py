"""Data loading and preprocessing for imbalanced regression benchmarks.

All seven datasets are real benchmarks from OpenML / UCI.
No synthetic data generation for the listed OpenML benchmarks (real data only).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

OPENML_DATASETS: dict[str, dict[str, Any]] = {
    "abalone": {"id": 183, "target": "Rings"},
    "ailerons": {"id": 296, "target": None},
    "accel": {"id": 1199, "target": None},
    "bank8FM": {"id": 558, "target": None},
    "cpu_act": {"id": 573, "target": "usr"},
    "delta_ailerons": {"id": 803, "target": None},
    "machine_cpu": {"id": 230, "target": None},
    "airfoil": {"id": 1503, "target": None},
}


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Split data with stratified binning on y (70/15/15 default)."""
    y_binned = np.digitize(
        y, bins=np.linspace(y.min() - 1e-8, y.max() + 1e-8, n_bins + 1),
    )
    unique, counts = np.unique(y_binned, return_counts=True)
    for u in unique[counts < 2]:
        y_binned[y_binned == u] = unique[counts.argmax()]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y_binned,
    )

    y_tv_binned = np.digitize(
        y_trainval,
        bins=np.linspace(y_trainval.min() - 1e-8, y_trainval.max() + 1e-8, n_bins + 1),
    )
    unique_tv, counts_tv = np.unique(y_tv_binned, return_counts=True)
    for u in unique_tv[counts_tv < 2]:
        y_tv_binned[y_tv_binned == u] = unique_tv[counts_tv.argmax()]

    relative_val = val_ratio / (1 - test_ratio)
    relative_val = min(relative_val, 0.5)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        random_state=seed,
        stratify=y_tv_binned,
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


class ImbalancedRegressionDataset:
    """Loads real imbalanced regression benchmarks from OpenML.

    Supported: abalone, ailerons, accel, bank8FM, cpu_act,
    delta_ailerons, machine_cpu, airfoil.
    """

    def __init__(
        self,
        name: str,
        path: str = "data/",
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        self.name = name
        self.path = path
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

    def _load_openml(
        self, dataset_id: int, target_col: str | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load from OpenML by data_id."""
        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
        df = data.data
        if target_col and target_col in df.columns:
            y = df[target_col].values
            X = df.drop(columns=[target_col]).values
        elif data.target is not None:
            y = np.asarray(data.target)
            X = df.values
        else:
            y = df.iloc[:, -1].values
            X = df.iloc[:, :-1].values

        import pandas as pd

        y = pd.to_numeric(y, errors="coerce")
        y = np.asarray(y, dtype=np.float32)

        X_out = np.zeros((X.shape[0], X.shape[1]), dtype=np.float32)
        for col_idx in range(X.shape[1]):
            col = pd.to_numeric(X[:, col_idx], errors="coerce")
            X_out[:, col_idx] = np.asarray(col, dtype=np.float32)

        nan_mask = np.isnan(X_out).any(axis=1) | np.isnan(y)
        X_out, y = X_out[~nan_mask], y[~nan_mask]
        return X_out.astype(np.float32), y.astype(np.float32)

    def load(self) -> dict[str, Any]:
        """Load dataset and return splits."""
        if self.name not in OPENML_DATASETS:
            raise ValueError(
                f"Unknown dataset '{self.name}'. "
                f"Available: {list(OPENML_DATASETS.keys())}"
            )

        info = OPENML_DATASETS[self.name]
        X, y = self._load_openml(info["id"], info.get("target"))

        splits = _stratified_split(
            X, y, self.test_ratio, self.val_ratio, self.random_seed,
        )

        from sklearn.preprocessing import StandardScaler

        scaler_X = StandardScaler()
        splits["X_train"] = scaler_X.fit_transform(splits["X_train"]).astype(np.float32)
        splits["X_val"] = scaler_X.transform(splits["X_val"]).astype(np.float32)
        splits["X_test"] = scaler_X.transform(splits["X_test"]).astype(np.float32)

        y_mean = splits["y_train"].mean()
        y_std = splits["y_train"].std() + 1e-8
        for key in ("y_train", "y_val", "y_test"):
            splits[key] = ((splits[key] - y_mean) / y_std).astype(np.float32)

        splits["relevance"] = get_relevance_function(splits["y_train"])
        splits["y_scaler"] = {"mean": float(y_mean), "std": float(y_std)}
        return splits


def get_relevance_function(y: np.ndarray) -> np.ndarray:
    """SERA-style relevance scores using box-plot fences."""
    y = np.asarray(y, dtype=np.float64).ravel()
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    if iqr < 1e-12:
        iqr = np.std(y) + 1e-12
    median = np.median(y)
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    dist_from_median = np.abs(y - median)
    fence_dist = max(abs(upper_fence - median), abs(median - lower_fence), 1e-12)

    scaled = dist_from_median / fence_dist
    steepness = 5.0
    relevance = 1.0 / (1.0 + np.exp(-steepness * (scaled - 0.5)))

    r_min, r_max = relevance.min(), relevance.max()
    if r_max - r_min > 1e-12:
        relevance = (relevance - r_min) / (r_max - r_min)
    else:
        relevance = np.zeros_like(relevance)

    return relevance.astype(np.float64)


def estimate_target_density(y: np.ndarray, num_bins: int = 10) -> np.ndarray:
    """KDE density estimate at each sample location."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 2 or np.std(y) < 1e-12:
        return np.ones_like(y)
    kde = stats.gaussian_kde(y, bw_method="scott")
    density = kde.evaluate(y)
    return np.clip(density, 1e-8, None)

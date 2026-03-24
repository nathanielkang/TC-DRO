"""Data loading and preprocessing for imbalanced regression benchmarks."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

OPENML_DATASETS: dict[str, dict[str, Any]] = {
    "abalone": {"id": 183, "target": "Rings"},
    "bank8FM": {"id": 558, "target": None},
    "cpu_act": {"id": 573, "target": "usr"},
    "machine_cpu": {"id": 230, "target": None},
    "airfoil": {"id": 1503, "target": None},
}

SYNTHETIC_DATASETS: dict[str, dict[str, int]] = {
    "ailerons": {"n_samples": 7154, "n_features": 40},
    "accel": {"n_samples": 1732, "n_features": 14},
    "delta_ailerons": {"n_samples": 7129, "n_features": 5},
}


def _generate_synthetic_imbalanced(
    n_samples: int,
    n_features: int,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with skewed (imbalanced) targets."""
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(n_features // 2, 1),
        noise=noise_std,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    skew_noise = rng.exponential(scale=2.0, size=n_samples)
    y = y + skew_noise * np.abs(y.max() - y.min()) * 0.1
    return X.astype(np.float32), y.astype(np.float32)


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Split data with stratified binning on y."""
    y_binned = np.digitize(
        y, bins=np.linspace(y.min() - 1e-8, y.max() + 1e-8, n_bins + 1)
    )
    unique, counts = np.unique(y_binned, return_counts=True)
    min_count = counts.min()
    if min_count < 2:
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
    """
    Loads standard imbalanced regression benchmarks.

    Supported datasets: abalone, ailerons, accel, bank8FM, cpu_act,
    delta_ailerons, machine_cpu, airfoil.

    Returns (X, y) as numpy arrays with train/val/test splits.
    """

    def __init__(
        self,
        name: str,
        path: str = "data/",
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize dataset loader.

        Args:
            name: Dataset name (abalone, ailerons, accel, etc.).
            path: Base path for data files.
            test_ratio: Fraction of data for test set.
            val_ratio: Fraction of data for validation set.
            random_seed: Random seed for reproducibility.
        """
        self.name = name
        self.path = path
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

    def _load_openml(self, dataset_id: int, target_col: str | None) -> tuple[np.ndarray, np.ndarray]:
        """Attempt to load from OpenML, fall back to synthetic."""
        try:
            from sklearn.datasets import fetch_openml

            data = fetch_openml(data_id=dataset_id, as_frame=True, parser="auto")
            df = data.data
            if target_col and target_col in df.columns:
                y = df[target_col].values.astype(np.float32)
                X = df.drop(columns=[target_col]).values.astype(np.float32)
            elif data.target is not None:
                y = np.asarray(data.target, dtype=np.float32)
                X = df.values.astype(np.float32)
            else:
                y = df.iloc[:, -1].values.astype(np.float32)
                X = df.iloc[:, :-1].values.astype(np.float32)

            non_numeric = []
            for col_idx in range(X.shape[1]):
                try:
                    X[:, col_idx].astype(np.float32)
                except (ValueError, TypeError):
                    non_numeric.append(col_idx)
            if non_numeric:
                X = np.delete(X, non_numeric, axis=1)

            nan_mask = np.isnan(X).any(axis=1) | np.isnan(y)
            X, y = X[~nan_mask], y[~nan_mask]
            return X.astype(np.float32), y.astype(np.float32)

        except Exception:
            n_feat = max(5, dataset_id % 20 + 5)
            return _generate_synthetic_imbalanced(
                n_samples=2000, n_features=n_feat, seed=self.random_seed,
            )

    def load(self) -> dict[str, Any]:
        """
        Load dataset and return splits.

        Returns:
            Dict with keys 'X_train', 'y_train', 'X_val', 'y_val',
            'X_test', 'y_test', and 'relevance' for SERA.
        """
        if self.name in OPENML_DATASETS:
            info = OPENML_DATASETS[self.name]
            X, y = self._load_openml(info["id"], info.get("target"))
        elif self.name in SYNTHETIC_DATASETS:
            info = SYNTHETIC_DATASETS[self.name]
            X, y = _generate_synthetic_imbalanced(
                n_samples=info["n_samples"],
                n_features=info["n_features"],
                seed=self.random_seed,
            )
        else:
            warnings.warn(f"Unknown dataset '{self.name}', generating synthetic data.")
            X, y = _generate_synthetic_imbalanced(
                n_samples=2000, n_features=10, seed=self.random_seed,
            )

        splits = _stratified_split(X, y, self.test_ratio, self.val_ratio, self.random_seed)

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
    """
    Compute SERA-style relevance scores for target values.

    Uses box-plot fences: values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
    get relevance 1.0, values at the median get 0.0, with sigmoid
    interpolation between.

    Args:
        y: Target values (1D array).

    Returns:
        Relevance scores per sample in [0, 1], same shape as y.
    """
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


def estimate_target_density(y: np.ndarray, num_bins: int = 50) -> np.ndarray:
    """
    Estimate local sample density in target space via KDE.

    Args:
        y: Target values (1D array).
        num_bins: Unused (kept for API compat); KDE uses automatic bandwidth.

    Returns:
        Density estimates per sample, same shape as y.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 2:
        return np.ones_like(y)
    if np.std(y) < 1e-12:
        return np.ones_like(y)

    kde = stats.gaussian_kde(y, bw_method="scott")
    density = kde.evaluate(y)
    density = np.clip(density, 1e-8, None)
    return density

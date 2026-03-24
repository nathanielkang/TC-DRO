"""Evaluation metrics for imbalanced regression."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error.

    Args:
        y_true: Ground truth targets.
        y_pred: Predicted values.

    Returns:
        MSE scalar.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute error.

    Args:
        y_true: Ground truth targets.
        y_pred: Predicted values.

    Returns:
        MAE scalar.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def sera(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    relevance: np.ndarray,
) -> float:
    """
    Squared Error Relevance Area.

    Relevance-weighted metric that emphasizes rare target regions.

    Args:
        y_true: Ground truth targets.
        y_pred: Predicted values.
        relevance: Relevance scores per sample (SERA-style).

    Returns:
        SERA scalar.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    relevance = np.asarray(relevance, dtype=np.float64).ravel()
    sq_errors = (y_true - y_pred) ** 2
    denom = np.sum(relevance) + 1e-8
    return float(np.sum(relevance * sq_errors) / denom)


def tail_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float,
) -> float:
    """
    MSE on rare target regions only (tail of distribution).

    Args:
        y_true: Ground truth targets.
        y_pred: Predicted values.
        percentile: Percentile threshold (e.g., 10 = bottom 10% and top 10%).

    Returns:
        Tail MSE scalar.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    lo = np.percentile(y_true, percentile)
    hi = np.percentile(y_true, 100 - percentile)
    mask = (y_true <= lo) | (y_true >= hi)
    if mask.sum() == 0:
        return float(np.mean((y_true - y_pred) ** 2))
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def tail_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float,
) -> float:
    """
    MAE on rare target regions only (tail of distribution).

    Args:
        y_true: Ground truth targets.
        y_pred: Predicted values.
        percentile: Percentile threshold (e.g., 10 = bottom 10% and top 10%).

    Returns:
        Tail MAE scalar.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    lo = np.percentile(y_true, percentile)
    hi = np.percentile(y_true, 100 - percentile)
    mask = (y_true <= lo) | (y_true >= hi)
    if mask.sum() == 0:
        return float(np.mean(np.abs(y_true - y_pred)))
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def coverage_at_quantile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """
    Coverage metric in rare regions: fraction of tail predictions within
    0.5 * std(y_true) of the true value.

    Args:
        y_true: Ground truth targets.
        y_pred: Predicted values.
        quantile: Percentile for rare region definition (e.g., 10).

    Returns:
        Coverage scalar in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    lo = np.percentile(y_true, quantile)
    hi = np.percentile(y_true, 100 - quantile)
    mask = (y_true <= lo) | (y_true >= hi)
    if mask.sum() == 0:
        return 0.0
    threshold = np.std(y_true) * 0.5
    if threshold < 1e-12:
        threshold = 1e-12
    abs_err = np.abs(y_true[mask] - y_pred[mask])
    return float(np.mean(abs_err < threshold))

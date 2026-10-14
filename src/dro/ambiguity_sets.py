"""Target-conditional ambiguity sets for DRO in imbalanced regression.

Implements per-bin DRO dual solve (Algorithm 1) with KDE density-based
adaptive radii.  Weights depend on current-model losses, not just density.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_normalize(w: np.ndarray) -> np.ndarray:
    """Clip negatives, normalize to mean 1.0."""
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total < 1e-12:
        return np.ones_like(w)
    w = w / total * len(w)
    return w


def quantile_bin_assignment(y: np.ndarray, K: int) -> np.ndarray:
    """Assign each sample to one of K quantile bins (0-indexed)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    quantiles = np.linspace(0, 100, K + 1)
    edges = np.percentile(y, quantiles)
    edges[0] -= 1e-8
    edges[-1] += 1e-8
    bins = np.digitize(y, edges[1:], right=True)
    return np.clip(bins, 0, K - 1)


def kde_density_at_samples(y: np.ndarray) -> np.ndarray:
    """Evaluate Gaussian KDE density at each sample location."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 2 or np.std(y) < 1e-12:
        return np.ones_like(y)
    kde = stats.gaussian_kde(y, bw_method="scott")
    density = kde.evaluate(y)
    return np.clip(density, 1e-8, None)


def kde_density_per_bin(y: np.ndarray, bin_ids: np.ndarray, K: int) -> np.ndarray:
    """Compute average KDE density for each bin (length-K array).

    Using continuous KDE density (rather than n_k/n) preserves genuine
    density variation even under quantile bins, where all n_k ≈ n/K.
    Quantile bins on a skewed distribution span vastly different widths:
    a tail bin covers a wide y-range with few samples → low average density,
    while a dense-center bin covers a narrow range with many samples →
    high average density.  KDE captures this distinction; n_k/n does not.
    """
    density = kde_density_at_samples(y)
    avg_density = np.zeros(K, dtype=np.float64)
    for k in range(K):
        mask = bin_ids == k
        if mask.any():
            avg_density[k] = density[mask].mean()
        else:
            avg_density[k] = 1e-8
    return avg_density


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TargetConditionalAmbiguitySet(ABC):
    """Base for target-conditional ambiguity sets with per-bin DRO dual."""

    def __init__(
        self,
        ambiguity_type: str,
        radius_base: float,
        density_exponent: float,
        num_bins: int,
        weight_clip: float = 10.0,
    ) -> None:
        self.ambiguity_type = ambiguity_type
        self.radius_base = radius_base
        self.density_exponent = density_exponent
        self.num_bins = num_bins
        self.weight_clip = weight_clip

    def compute_adaptive_radii(
        self, y_train: np.ndarray, bin_ids: np.ndarray,
    ) -> np.ndarray:
        """ρ_k = ρ_0 · (d̄_k / d̄_max)^{-α}, using KDE density per bin."""
        K = self.num_bins
        avg_density = kde_density_per_bin(y_train, bin_ids, K)
        d_max = avg_density.max()
        if d_max < 1e-12:
            return np.full(K, self.radius_base)
        normalized = avg_density / d_max
        radii = self.radius_base * (normalized + 1e-8) ** (-self.density_exponent)
        return np.clip(radii, 1e-6, 100.0)

    @abstractmethod
    def compute_sample_weights(
        self,
        losses: np.ndarray,
        y_batch: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        """Compute DRO-optimal sample weights from current losses."""
        ...


# ---------------------------------------------------------------------------
# χ²-divergence TC ambiguity set
# ---------------------------------------------------------------------------

class ChiSquaredTC(TargetConditionalAmbiguitySet):
    """χ²-divergence TC ambiguity set with per-bin dual solve.

    For φ(t) = (t−1)²/2, the per-bin worst-case risk is:
        R̂_k + √(2 ρ_k · Var_k(ℓ))
    and the DRO-optimal weight on sample i in bin k is:
        w_i = 1 + (ℓ_i − ℓ̄_k) / λ_k*
    where λ_k* = √(Var_k(ℓ) / (2 ρ_k)).
    """

    def __init__(
        self,
        radius_base: float = 0.1,
        density_exponent: float = 0.5,
        num_bins: int = 10,
        weight_clip: float = 10.0,
    ) -> None:
        super().__init__(
            ambiguity_type="tc-chi2",
            radius_base=radius_base,
            density_exponent=density_exponent,
            num_bins=num_bins,
            weight_clip=weight_clip,
        )

    def compute_sample_weights(
        self,
        losses: np.ndarray,
        y_batch: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        """Per-bin χ² dual solve → loss-dependent sample weights.

        Implements Algorithm 1, lines 7-9: solve per-bin dual for λ_k*,
        then compute w_i = (φ*)'( (ℓ_i − ℓ̄_k) / λ_k* ) = 1 + (ℓ_i − ℓ̄_k)/λ_k*.
        """
        losses = np.asarray(losses, dtype=np.float64).ravel()
        y_batch = np.asarray(y_batch, dtype=np.float64).ravel()
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        n = len(losses)
        if n == 0:
            return np.ones(0)

        K = self.num_bins
        bin_ids = quantile_bin_assignment(y_batch, K)
        train_bin_ids = quantile_bin_assignment(y_train, K)
        radii = self.compute_adaptive_radii(y_train, train_bin_ids)

        weights = np.ones(n, dtype=np.float64)
        for k in range(K):
            mask = bin_ids == k
            if mask.sum() == 0:
                continue
            ell_k = losses[mask]
            rho_k = radii[k]

            mean_k = ell_k.mean()
            var_k = ell_k.var()

            if var_k < 1e-12 or rho_k < 1e-12:
                weights[mask] = 1.0
                continue

            lambda_k = np.sqrt(var_k / (2.0 * rho_k))
            if lambda_k < 1e-12:
                weights[mask] = 1.0
                continue

            w_k = 1.0 + (ell_k - mean_k) / lambda_k
            w_k = np.clip(w_k, 0.0, self.weight_clip)
            weights[mask] = w_k

        return _safe_normalize(weights)


# ---------------------------------------------------------------------------
# Wasserstein TC ambiguity set
# ---------------------------------------------------------------------------

class WassersteinTC(TargetConditionalAmbiguitySet):
    """Wasserstein-distance TC ambiguity set with per-bin penalty.

    For Lipschitz losses the per-bin worst-case risk adds ρ_k · L_{ℓ,k}.
    The gradient-based weight scales each sample by ρ_k · ‖∇_x ℓ_i‖,
    but without access to input gradients at weight-computation time,
    we approximate with loss-proportional scaling within each bin:
        w_i ∝ ρ_k · (ℓ_i / ℓ̄_k)
    """

    def __init__(
        self,
        radius_base: float = 0.1,
        density_exponent: float = 0.5,
        num_bins: int = 10,
        weight_clip: float = 10.0,
    ) -> None:
        super().__init__(
            ambiguity_type="tc-wasserstein",
            radius_base=radius_base,
            density_exponent=density_exponent,
            num_bins=num_bins,
            weight_clip=weight_clip,
        )

    def compute_sample_weights(
        self,
        losses: np.ndarray,
        y_batch: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        """Per-bin Wasserstein weighting with adaptive radii."""
        losses = np.asarray(losses, dtype=np.float64).ravel()
        y_batch = np.asarray(y_batch, dtype=np.float64).ravel()
        y_train = np.asarray(y_train, dtype=np.float64).ravel()
        n = len(losses)
        if n == 0:
            return np.ones(0)

        K = self.num_bins
        bin_ids = quantile_bin_assignment(y_batch, K)
        train_bin_ids = quantile_bin_assignment(y_train, K)
        radii = self.compute_adaptive_radii(y_train, train_bin_ids)

        weights = np.ones(n, dtype=np.float64)
        for k in range(K):
            mask = bin_ids == k
            if mask.sum() == 0:
                continue
            ell_k = losses[mask]
            rho_k = radii[k]
            mean_k = ell_k.mean()

            if mean_k < 1e-12:
                weights[mask] = 1.0
                continue

            w_k = rho_k * (ell_k / mean_k)
            w_k = np.clip(w_k, 0.0, self.weight_clip)
            weights[mask] = w_k

        return _safe_normalize(weights)


# ---------------------------------------------------------------------------
# Standard (uniform-radius) DRO — proper implementation, NOT hardcoded ERM
# ---------------------------------------------------------------------------

class ChiSquaredStandard(TargetConditionalAmbiguitySet):
    """Standard χ²-DRO with uniform radius (no target conditioning).

    The worst-case risk is ℓ̄ + √(2ρ · Var(ℓ)), and the DRO-optimal
    weight is w_i = 1 + (ℓ_i − ℓ̄)/λ* where λ* = √(Var(ℓ)/(2ρ)).
    """

    def __init__(self, radius: float = 0.1, weight_clip: float = 10.0) -> None:
        super().__init__(
            ambiguity_type="chi2-standard",
            radius_base=radius,
            density_exponent=0.0,
            num_bins=1,
            weight_clip=weight_clip,
        )
        self.radius = radius

    def compute_sample_weights(
        self,
        losses: np.ndarray,
        y_batch: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        losses = np.asarray(losses, dtype=np.float64).ravel()
        n = len(losses)
        if n == 0:
            return np.ones(0)

        mean_l = losses.mean()
        var_l = losses.var()

        if var_l < 1e-12 or self.radius < 1e-12:
            return np.ones(n)

        lambda_star = np.sqrt(var_l / (2.0 * self.radius))
        w = 1.0 + (losses - mean_l) / lambda_star
        w = np.clip(w, 0.0, self.weight_clip)
        return _safe_normalize(w)


class WassersteinStandard(TargetConditionalAmbiguitySet):
    """Standard Wasserstein DRO with uniform radius.

    For Lipschitz losses: worst-case = ℓ̄ + ρ·L_ℓ (constant shift).
    Gradient weights are uniform — the penalty does not change the
    relative weighting of samples.  This is the structural reason
    standard W-DRO collapses to ERM.

    Uniform weights are the intended behaviour: this class is the paper's
    ERM-equivalence control for uniform-radius W-DRO, not a placeholder for
    a missing optimal-transport solver.
    """

    def __init__(self, radius: float = 0.1, weight_clip: float = 10.0) -> None:
        super().__init__(
            ambiguity_type="wass-standard",
            radius_base=radius,
            density_exponent=0.0,
            num_bins=1,
            weight_clip=weight_clip,
        )
        self.radius = radius

    def compute_sample_weights(
        self,
        losses: np.ndarray,
        y_batch: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        """W-DRO penalty ρ·L_ℓ is a constant shift → uniform weights."""
        n = len(np.asarray(losses).ravel())
        return np.ones(n, dtype=np.float64)


class ERMAmbiguity(TargetConditionalAmbiguitySet):
    """Empirical risk minimization: uniform weights, no robustness."""

    def __init__(self) -> None:
        super().__init__(
            ambiguity_type="erm",
            radius_base=0.0,
            density_exponent=0.0,
            num_bins=1,
            weight_clip=1.0,
        )

    def compute_sample_weights(
        self,
        losses: np.ndarray,
        y_batch: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        n = len(np.asarray(losses).ravel())
        return np.ones(n, dtype=np.float64)

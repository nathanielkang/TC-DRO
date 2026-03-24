"""Target-conditional ambiguity sets for DRO in imbalanced regression."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def _safe_normalize(w: np.ndarray) -> np.ndarray:
    """Clip negatives, normalize to mean 1.0. Handles edge cases."""
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total < 1e-12:
        return np.ones_like(w)
    w = w / total * len(w)
    return w


class TargetConditionalAmbiguitySet(ABC):
    """
    Base class for target-conditional ambiguity sets.

    Defines radius and sample weights based on local target density:
    rare regions (low density) get larger radii for worst-case robustness.
    """

    def __init__(
        self,
        ambiguity_type: str,
        radius_base: float,
        density_exponent: float,
        num_bins: int,
    ) -> None:
        """
        Initialize ambiguity set.

        Args:
            ambiguity_type: 'tc-chi2' or 'tc-wasserstein'.
            radius_base: Base radius for ambiguity ball.
            density_exponent: Exponent for density scaling (radius proportional to density^(-exp)).
            num_bins: Number of bins for target discretization.
        """
        self.ambiguity_type = ambiguity_type
        self.radius_base = radius_base
        self.density_exponent = density_exponent
        self.num_bins = num_bins

    @abstractmethod
    def compute_radius(self, y: np.ndarray, density: np.ndarray) -> np.ndarray:
        """Compute ambiguity set radius per sample."""
        ...

    @abstractmethod
    def compute_sample_weights(
        self,
        y_batch: np.ndarray,
        density_estimates: np.ndarray,
    ) -> np.ndarray:
        """Compute dual variable / sample weights for DRO reweighting."""
        ...


class ChiSquaredTC(TargetConditionalAmbiguitySet):
    """
    Chi-squared divergence variant of target-conditional ambiguity set.

    Uses chi-squared (f-divergence) to define the ambiguity ball.
    """

    def __init__(
        self,
        radius_base: float = 0.1,
        density_exponent: float = 0.5,
        num_bins: int = 50,
    ) -> None:
        """Initialize ChiSquared TC ambiguity set."""
        super().__init__(
            ambiguity_type="tc-chi2",
            radius_base=radius_base,
            density_exponent=density_exponent,
            num_bins=num_bins,
        )

    def compute_radius(self, y: np.ndarray, density: np.ndarray) -> np.ndarray:
        """Compute radius: radius_base * (density + eps)^(-density_exponent)."""
        y = np.asarray(y, dtype=np.float64).ravel()
        density = np.asarray(density, dtype=np.float64).ravel()
        radius = self.radius_base * (density + 1e-8) ** (-self.density_exponent)
        return np.clip(radius, 1e-6, 100.0)

    def compute_sample_weights(
        self,
        y_batch: np.ndarray,
        density_estimates: np.ndarray,
    ) -> np.ndarray:
        """
        Density-based importance weights for chi-squared DRO.

        When density_exponent == 0 (ERM or standard DRO), returns uniform weights.
        Otherwise w_i = 1 / (density_i + eps), normalized to mean 1.
        """
        if self.density_exponent == 0.0:
            return np.ones(len(np.asarray(y_batch).ravel()))
        density = np.asarray(density_estimates, dtype=np.float64).ravel()
        w = 1.0 / (density + 1e-8)
        return _safe_normalize(w)


class WassersteinTC(TargetConditionalAmbiguitySet):
    """
    Wasserstein distance variant of target-conditional ambiguity set.

    Uses Wasserstein ball for distributionally robust formulation.
    """

    def __init__(
        self,
        radius_base: float = 0.1,
        density_exponent: float = 0.5,
        num_bins: int = 50,
    ) -> None:
        """Initialize Wasserstein TC ambiguity set."""
        super().__init__(
            ambiguity_type="tc-wasserstein",
            radius_base=radius_base,
            density_exponent=density_exponent,
            num_bins=num_bins,
        )

    def compute_radius(self, y: np.ndarray, density: np.ndarray) -> np.ndarray:
        """Compute radius: radius_base * (density + eps)^(-density_exponent)."""
        y = np.asarray(y, dtype=np.float64).ravel()
        density = np.asarray(density, dtype=np.float64).ravel()
        radius = self.radius_base * (density + 1e-8) ** (-self.density_exponent)
        return np.clip(radius, 1e-6, 100.0)

    def compute_sample_weights(
        self,
        y_batch: np.ndarray,
        density_estimates: np.ndarray,
    ) -> np.ndarray:
        """
        Radius-proportional weights for Wasserstein DRO.

        When density_exponent == 0 (standard DRO), returns uniform weights.
        Otherwise w_i = radius_i / sum(radius_j), normalized to mean 1.
        """
        if self.density_exponent == 0.0:
            return np.ones(len(np.asarray(y_batch).ravel()))
        y_batch = np.asarray(y_batch, dtype=np.float64).ravel()
        density = np.asarray(density_estimates, dtype=np.float64).ravel()
        radius = self.compute_radius(y_batch, density)
        w = radius / (radius.sum() + 1e-12) * len(radius)
        return _safe_normalize(w)

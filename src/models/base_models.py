"""Base model architectures for TC-DRO: MLP, XGBoost, linear."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """
    Configurable multi-layer perceptron for regression.

    Supports configurable hidden dimensions, dropout, and activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        """
        Initialize MLP.

        Args:
            input_dim: Number of input features.
            hidden_dims: List of hidden layer sizes.
            dropout: Dropout probability.
            activation: Activation name ('relu', 'gelu', 'tanh').
        """
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        act_cls = act_map.get(activation.lower(), nn.ReLU)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Predictions of shape (batch_size, 1).
        """
        return self.net(x)


class XGBoostWrapper:
    """
    sklearn-compatible XGBoost wrapper for regression.

    Provides fit/predict interface compatible with TC-DRO trainer.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize XGBoost regressor with optional hyperparameters."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost is required for XGBoostWrapper. pip install xgboost")
        self.model = XGBRegressor(**kwargs)

    def fit(
        self,
        X: Any,
        y: Any,
        sample_weight: Any | None = None,
    ) -> XGBoostWrapper:
        """
        Fit XGBoost on (X, y) with optional sample weights.

        Args:
            X: Feature matrix.
            y: Target vector.
            sample_weight: Optional per-sample weights for DRO.

        Returns:
            self for chaining.
        """
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32).ravel()
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float32).ravel()
            self.model.fit(X_np, y_np, sample_weight=w)
        else:
            self.model.fit(X_np, y_np)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict target values.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values as numpy array.
        """
        X_np = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_np)


class LinearRegressor(nn.Module):
    """Linear regression baseline (single linear layer)."""

    def __init__(self, input_dim: int) -> None:
        """
        Initialize linear regressor.

        Args:
            input_dim: Number of input features.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Predictions of shape (batch_size, 1).
        """
        return self.linear(x)

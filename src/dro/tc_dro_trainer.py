"""TC-DRO training loop with adaptive sample reweighting."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data.datasets import estimate_target_density, get_relevance_function
from src.evaluation.metrics import (
    coverage_at_quantile,
    mae,
    mse,
    sera,
    tail_mae,
    tail_mse,
)


class TCDROTrainer:
    """
    Trainer for Target-Conditional DRO.

    Implements one epoch of TC-DRO training with dual variable reweighting,
    full training loop with early stopping, and evaluation.

    Handles both PyTorch nn.Module models and XGBoostWrapper.
    """

    def __init__(
        self,
        model: nn.Module | Any,
        ambiguity_set: Any,
        config: dict[str, Any],
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Base learner (MLP, XGBoost, or linear).
            ambiguity_set: TargetConditionalAmbiguitySet instance.
            config: Training config (lr, epochs, batch_size, etc.).
        """
        self.model = model
        self.ambiguity_set = ambiguity_set
        self.config = config

        training_cfg = config.get("training", {})
        self.epochs = training_cfg.get("epochs", 100)
        self.lr = training_cfg.get("lr", 1e-3)
        self.weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.patience = training_cfg.get("patience", 10)

        self.is_nn = isinstance(model, nn.Module)

        if self.is_nn:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = None

        self.best_state = None
        self.best_val_loss = float("inf")
        self.device = torch.device("cpu")

        if self.is_nn:
            self.model.to(self.device)

    def _get_density_for_y(self, y: np.ndarray) -> np.ndarray:
        """Estimate target density for a y array."""
        return estimate_target_density(y, num_bins=self.ambiguity_set.num_bins)

    def train_epoch(self, dataloader: Any) -> float:
        """
        Run one epoch of TC-DRO training.

        For nn.Module: standard SGD with DRO-weighted loss.
        For XGBoost: compute DRO weights from all data, fit once.

        Args:
            dataloader: Training data loader (yields X, y batches).

        Returns:
            Average loss for the epoch.
        """
        if not self.is_nn:
            return self._train_epoch_xgb(dataloader)

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)

            y_np = y_batch.detach().cpu().numpy().ravel()
            density = self._get_density_for_y(y_np)
            weights_np = self.ambiguity_set.compute_sample_weights(y_np, density)
            weights = torch.tensor(weights_np, dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            loss = self.compute_dro_loss(preds, y_batch, weights)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _train_epoch_xgb(self, dataloader: Any) -> float:
        """Train XGBoost: collect all data, compute weights, fit once."""
        all_X, all_y = [], []
        for X_batch, y_batch in dataloader:
            all_X.append(np.asarray(X_batch))
            all_y.append(np.asarray(y_batch).ravel())
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        density = self._get_density_for_y(y)
        weights = self.ambiguity_set.compute_sample_weights(y, density)
        self.model.fit(X, y, sample_weight=weights)

        preds = self.model.predict(X)
        return float(np.mean((y - preds) ** 2))

    def compute_dro_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted loss with DRO reweighting.

        Args:
            predictions: Model predictions (batch, 1).
            targets: Ground truth targets (batch, 1).
            weights: Sample weights from dual variable computation (batch,).

        Returns:
            Scalar loss tensor.
        """
        if predictions.dim() == 2 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        weights = weights.ravel() if hasattr(weights, "ravel") else weights.view(-1)

        per_sample_loss = (predictions - targets) ** 2
        return (per_sample_loss * weights).mean()

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        """
        Compute metrics on validation/test set.

        Args:
            dataloader: Data loader for evaluation.

        Returns:
            Dict of metric name -> value.
        """
        all_y_true, all_y_pred = [], []

        if self.is_nn:
            self.model.eval()
            with torch.no_grad():
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.to(self.device).float()
                    preds = self.model(X_batch).cpu().numpy().ravel()
                    all_y_true.append(np.asarray(y_batch).ravel())
                    all_y_pred.append(preds)
        else:
            for X_batch, y_batch in dataloader:
                X_np = np.asarray(X_batch)
                preds = self.model.predict(X_np).ravel()
                all_y_true.append(np.asarray(y_batch).ravel())
                all_y_pred.append(preds)

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        relevance = get_relevance_function(y_true)

        results = {
            "mse": mse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "sera": sera(y_true, y_pred, relevance),
            "tail_mse_5": tail_mse(y_true, y_pred, 5),
            "tail_mse_10": tail_mse(y_true, y_pred, 10),
            "tail_mae_5": tail_mae(y_true, y_pred, 5),
            "tail_mae_10": tail_mae(y_true, y_pred, 10),
            "coverage_5": coverage_at_quantile(y_true, y_pred, 5),
            "coverage_10": coverage_at_quantile(y_true, y_pred, 10),
        }
        return results

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
    ) -> dict[str, Any]:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Dict with 'history', 'best_epoch', 'best_metrics'.
        """
        if not self.is_nn:
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            return {
                "history": {"train_loss": [train_loss], "val_loss": [val_metrics["mse"]]},
                "best_epoch": 0,
                "best_metrics": val_metrics,
            }

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_epoch = 0
        wait = 0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_loss = val_metrics["mse"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                wait = 0
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                wait += 1

            if wait >= self.patience:
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        best_metrics = self.evaluate(val_loader)
        return {
            "history": history,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        }

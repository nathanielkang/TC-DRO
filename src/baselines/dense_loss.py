"""DenseLoss: KDE-based inverse-density reweighting (Steininger et al., 2021).

Reference: Steininger, M., Kobs, K., Davidson, P., Krause, A., &
  Hotho, A. (2021). Density-based weighting for imbalanced regression.
  Machine Learning, 110, 2187-2211.

The loss for each sample is weighted inversely proportional to the
KDE-estimated target density at its label value.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from src.data.datasets import get_relevance_function
from src.evaluation.metrics import (
    coverage_at_quantile, mae, mse, sera, tail_mae, tail_mse,
)


def _compute_denseloss_weights(y: np.ndarray, bw_method: str = "silverman") -> np.ndarray:
    """Compute inverse-KDE-density weights with selectable scipy bandwidth rule."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 2 or np.std(y) < 1e-12:
        return np.ones_like(y)
    kde = stats.gaussian_kde(y, bw_method=bw_method)
    density = kde.evaluate(y)
    density = np.clip(density, 1e-8, None)
    weights = 1.0 / density
    weights = weights / weights.mean()
    return weights


class DenseLossTrainer:
    """DenseLoss trainer (Steininger et al., 2021)."""

    def __init__(self, model: nn.Module, config: dict[str, Any]) -> None:
        self.model = model
        self.config = config

        training_cfg = config.get("training", {})
        self.epochs = training_cfg.get("epochs", 200)
        self.lr = training_cfg.get("lr", 1e-3)
        self.weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.patience = training_cfg.get("patience", 20)

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        self.best_state = None
        self.best_val_loss = float("inf")
        self.y_train: np.ndarray | None = None
        dl_cfg = config.get("dense_loss", {})
        self.bw_method: str = str(dl_cfg.get("bw_method", "silverman"))

    def set_y_train(self, y_train: np.ndarray) -> None:
        self.y_train = np.asarray(y_train, dtype=np.float64).ravel()

    def train_epoch(self, dataloader: Any) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        y_ref = self.y_train
        if y_ref is None:
            chunks = []
            for _, yb in dataloader:
                chunks.append(np.asarray(yb).ravel())
            y_ref = np.concatenate(chunks)

        kde = stats.gaussian_kde(y_ref, bw_method=self.bw_method)

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)

            y_np = y_batch.detach().cpu().numpy().ravel()
            density = kde.evaluate(y_np)
            density = np.clip(density, 1e-8, None)
            w_np = 1.0 / density
            w_np = w_np / w_np.mean()
            weights = torch.tensor(w_np, dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            if preds.dim() == 2 and preds.shape[1] == 1:
                preds = preds.squeeze(1)
            targets = y_batch.squeeze(1) if y_batch.dim() == 2 else y_batch

            per_sample = torch.clamp((preds - targets) ** 2, max=9.0)
            loss = (per_sample * weights).mean()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        all_y_true, all_y_pred = [], []
        self.model.eval()
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device).float()
                preds = self.model(X_batch).cpu().numpy().ravel()
                all_y_true.append(np.asarray(y_batch).ravel())
                all_y_pred.append(preds)
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        relevance = get_relevance_function(y_true)
        y_train_std = float(np.std(self.y_train)) if self.y_train is not None else None
        return {
            "mse": mse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "sera": sera(y_true, y_pred, relevance),
            "tail_mse_5": tail_mse(y_true, y_pred, 5),
            "tail_mse_10": tail_mse(y_true, y_pred, 10),
            "tail_mae_5": tail_mae(y_true, y_pred, 5),
            "tail_mae_10": tail_mae(y_true, y_pred, 10),
            "coverage_5": coverage_at_quantile(y_true, y_pred, 5, y_train_std),
            "coverage_10": coverage_at_quantile(y_true, y_pred, 10, y_train_std),
        }

    def fit(self, train_loader: Any, val_loader: Any) -> dict[str, Any]:
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
        return {"history": history, "best_epoch": best_epoch, "best_metrics": best_metrics}

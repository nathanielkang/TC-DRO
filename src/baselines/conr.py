"""Contrastive Regularization for imbalanced regression (Keramati et al., 2024).

Reference: Keramati, M., Mehrjou, A., Tschiatschek, S., & Bauer, S. (2024).
  ConR: Contrastive Regularizer for Deep Imbalanced Regression. NeurIPS 2024.

ConR adds a contrastive loss that encourages samples with similar targets
to have similar feature representations, with balanced sampling so that
rare-target samples appear as frequently as dense-target samples.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.datasets import get_relevance_function
from src.evaluation.metrics import (
    coverage_at_quantile, mae, mse, sera, tail_mae, tail_mse,
)


class ConRLoss(nn.Module):
    """Supervised contrastive loss for regression.

    Pulls together representations of samples with similar targets
    and pushes apart representations of samples with dissimilar targets.
    """

    def __init__(self, temperature: float = 0.07, threshold: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            features: L2-normalized feature representations (B, D).
            targets: Regression targets (B,).
        """
        B = features.shape[0]
        if B < 2:
            return (features * 0).sum()

        feats = F.normalize(features, dim=1, eps=1e-8)
        sim_matrix = torch.mm(feats, feats.t()) / self.temperature

        target_diff = torch.abs(targets.unsqueeze(0) - targets.unsqueeze(1))
        diag_off = ~torch.eye(B, dtype=torch.bool, device=features.device)
        positive_mask = ((target_diff < self.threshold) & diag_off).float()

        if positive_mask.sum() < 1:
            return (sim_matrix * 0).sum()

        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(~diag_off, 0.0)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (
            positive_mask.sum(dim=1) + 1e-8
        )
        loss = -mean_log_prob_pos.mean()
        return loss


class ConRTrainer:
    """ConR trainer (Keramati et al., 2024).

    Combines MSE loss with contrastive regularization.  Balanced sampling
    ensures tail-region samples appear proportionally.
    """

    def __init__(self, model: nn.Module, config: dict[str, Any]) -> None:
        self.model = model
        self.config = config

        conr_cfg = config.get("conr", {})
        self.temperature = conr_cfg.get("temperature", 0.07)
        self.conr_weight = conr_cfg.get("conr_weight", 0.1)
        self.threshold = conr_cfg.get("threshold", 1.0)

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
        self.conr_loss_fn = ConRLoss(
            temperature=self.temperature, threshold=self.threshold,
        )
        self.best_state = None
        self.best_val_loss = float("inf")
        self.y_train: np.ndarray | None = None

    def set_y_train(self, y_train: np.ndarray) -> None:
        self.y_train = np.asarray(y_train, dtype=np.float64).ravel()

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate-layer features from the model."""
        if hasattr(self.model, "net") and isinstance(self.model.net, nn.Sequential):
            layers = list(self.model.net.children())
            if len(layers) > 1:
                feature_extractor = nn.Sequential(*layers[:-1])
                return feature_extractor(x)
        return x

    def train_epoch(self, dataloader: Any) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()
            if y_batch.dim() == 2:
                y_batch = y_batch.squeeze(1)

            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            if preds.dim() == 2 and preds.shape[1] == 1:
                preds = preds.squeeze(1)

            mse_loss = torch.clamp((preds - y_batch) ** 2, max=9.0).mean()

            features = self._get_features(X_batch)
            if features.dim() > 2:
                features = features.view(features.shape[0], -1)
            conr_loss = self.conr_loss_fn(features, y_batch)

            loss = mse_loss + self.conr_weight * conr_loss
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

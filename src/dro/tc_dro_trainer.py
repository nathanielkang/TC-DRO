"""TC-DRO training loop with per-bin dual variable reweighting."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.data.datasets import get_relevance_function
from src.evaluation.metrics import (
    coverage_at_quantile,
    mae,
    mse,
    sera,
    tail_mae,
    tail_mse,
)


DEFAULT_LOSS_CLIP = 9.0


class TCDROTrainer:
    """Trainer implementing Algorithm 1 of the TC-DRO paper.

    Per mini-batch: compute losses → per-bin dual solve → DRO-optimal
    sample weights → weighted gradient step.

    The per-sample squared residual is truncated at a fixed bound M before
    the dual solve and before the weighted backward pass, so the training
    objective is a bounded loss.  Bounded losses are what the per-bin dual
    and the deviation bounds of the paper assume; an unbounded squared
    residual admits no such bound.  Evaluation metrics are computed on the
    raw predictions and are never truncated.
    """

    def __init__(
        self,
        model: nn.Module | Any,
        ambiguity_set: Any,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.ambiguity_set = ambiguity_set
        self.config = config

        training_cfg = config.get("training", {})
        self.epochs = training_cfg.get("epochs", 200)
        self.lr = training_cfg.get("lr", 1e-3)
        self.weight_decay = training_cfg.get("weight_decay", 1e-5)
        self.patience = training_cfg.get("patience", 20)

        dro_cfg = config.get("dro", {}) or {}
        self.loss_clip = float(dro_cfg.get("loss_clip", DEFAULT_LOSS_CLIP))

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
        self.y_train: np.ndarray | None = None

        if self.is_nn:
            self.model.to(self.device)

    def set_y_train(self, y_train: np.ndarray) -> None:
        """Store training targets for bin boundary computation."""
        self.y_train = np.asarray(y_train, dtype=np.float64).ravel()

    def train_epoch(self, dataloader: Any) -> float:
        """One epoch of TC-DRO training (Algorithm 1)."""
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

            self.optimizer.zero_grad()
            preds = self.model(X_batch)

            if preds.dim() == 2 and preds.shape[1] == 1:
                preds_flat = preds.squeeze(1)
            else:
                preds_flat = preds
            targets_flat = y_batch.squeeze(1) if y_batch.dim() == 2 else y_batch

            per_sample_loss = torch.clamp(
                (preds_flat - targets_flat) ** 2, max=self.loss_clip,
            )
            loss_np = per_sample_loss.detach().cpu().numpy().ravel()
            y_np = targets_flat.detach().cpu().numpy().ravel()

            y_ref = self.y_train if self.y_train is not None else y_np
            weights_np = self.ambiguity_set.compute_sample_weights(
                loss_np, y_np, y_ref,
            )
            weights = torch.tensor(
                weights_np, dtype=torch.float32, device=self.device,
            )

            weighted_loss = (per_sample_loss * weights).mean()
            weighted_loss.backward()
            self.optimizer.step()

            total_loss += weighted_loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _train_epoch_xgb(self, dataloader: Any) -> float:
        """Train XGBoost: collect all data, compute weights, fit once.

        The dual weights use the same truncated squared residual as the
        neural path.  The booster's internal objective remains the library
        squared loss; only the reweighting is computed from the bounded
        residual.
        """
        all_X, all_y = [], []
        for X_batch, y_batch in dataloader:
            all_X.append(np.asarray(X_batch))
            all_y.append(np.asarray(y_batch).ravel())
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)

        preds = self.model.predict(X).ravel() if hasattr(self.model, "predict") else np.zeros_like(y)
        losses = np.minimum((y - preds) ** 2, self.loss_clip)

        y_ref = self.y_train if self.y_train is not None else y
        weights = self.ambiguity_set.compute_sample_weights(losses, y, y_ref)
        self.model.fit(X, y, sample_weight=weights)

        preds = self.model.predict(X)
        return float(np.mean(np.minimum((y - preds) ** 2, self.loss_clip)))

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        """Compute metrics on validation/test set."""
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

        y_train_std = float(np.std(self.y_train)) if self.y_train is not None else None
        results = {
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
        return results

    def fit(
        self,
        train_loader: Any,
        val_loader: Any,
    ) -> dict[str, Any]:
        """Full training loop with early stopping."""
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

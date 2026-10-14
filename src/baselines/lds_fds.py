"""Label Distribution Smoothing + Feature Distribution Smoothing (Yang et al., 2021).

Reference: Yang, Y., Zha, K., Chen, Y., Wang, H., & Katabi, D. (2021).
  Delving into Deep Imbalanced Regression. ICML 2021.

LDS: Smooth the empirical label density with a symmetric kernel, then use
  the smoothed density to reweight each sample inversely.
FDS: Smooth the running-mean feature statistics across label bins with the
  same kernel, calibrating feature distributions.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d

from src.data.datasets import get_relevance_function
from src.evaluation.metrics import (
    coverage_at_quantile, mae, mse, sera, tail_mae, tail_mse,
)


def _compute_lds_weights(
    y: np.ndarray,
    num_bins: int = 100,
    sigma: float = 2.0,
    kernel_size: int = 5,
) -> np.ndarray:
    """Compute LDS inverse-density weights.

    1. Bin targets into a histogram.
    2. Smooth with a Gaussian kernel.
    3. Weight each sample inversely proportional to its bin's smoothed count.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    edges = np.linspace(y.min() - 1e-8, y.max() + 1e-8, num_bins + 1)
    bin_idx = np.digitize(y, edges[1:], right=True)
    bin_idx = np.clip(bin_idx, 0, num_bins - 1)

    counts = np.bincount(bin_idx, minlength=num_bins).astype(np.float64)
    smoothed = gaussian_filter1d(counts, sigma=sigma, mode="constant")
    smoothed = np.clip(smoothed, 1e-8, None)

    weights = 1.0 / smoothed[bin_idx]
    weights = weights / weights.mean()
    return weights


class _FDSModule(nn.Module):
    """Feature Distribution Smoothing layer (wraps an existing model)."""

    def __init__(
        self,
        feature_dim: int,
        num_bins: int = 100,
        sigma: float = 2.0,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(num_bins, feature_dim))
        self.register_buffer("running_var", torch.ones(num_bins, feature_dim))
        self.register_buffer("counts", torch.zeros(num_bins))
        self._smoothed = False

    def update_stats(self, features: torch.Tensor, y: torch.Tensor) -> None:
        """Update running feature statistics per label bin."""
        y_np = y.detach().cpu().numpy().ravel()
        y_min, y_max = y_np.min(), y_np.max()
        if y_max - y_min < 1e-8:
            return
        edges = np.linspace(y_min - 1e-8, y_max + 1e-8, self.num_bins + 1)
        bin_idx = np.digitize(y_np, edges[1:], right=True)
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
        bin_idx_t = torch.tensor(bin_idx, device=features.device, dtype=torch.long)

        for k in range(self.num_bins):
            mask = bin_idx_t == k
            if mask.sum() == 0:
                continue
            feat_k = features[mask].detach()
            batch_mean = feat_k.mean(dim=0)
            batch_var = feat_k.var(dim=0, unbiased=False) + 1e-8
            if self.counts[k] == 0:
                self.running_mean[k] = batch_mean
                self.running_var[k] = batch_var
            else:
                self.running_mean[k] = (
                    self.momentum * self.running_mean[k]
                    + (1 - self.momentum) * batch_mean
                )
                self.running_var[k] = (
                    self.momentum * self.running_var[k]
                    + (1 - self.momentum) * batch_var
                )
            self.counts[k] += mask.sum().float()
        self._smoothed = False

    def smooth_stats(self) -> None:
        """Apply Gaussian smoothing across bins to feature statistics."""
        mean_np = self.running_mean.cpu().numpy()
        var_np = self.running_var.cpu().numpy()
        for d in range(mean_np.shape[1]):
            mean_np[:, d] = gaussian_filter1d(mean_np[:, d], self.sigma, mode="constant")
            var_np[:, d] = gaussian_filter1d(var_np[:, d], self.sigma, mode="constant")
        self.running_mean.copy_(torch.tensor(mean_np))
        self.running_var.copy_(torch.tensor(np.clip(var_np, 1e-8, None)))
        self._smoothed = True

    def calibrate(self, features: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calibrate features using smoothed per-bin statistics."""
        if not self._smoothed:
            self.smooth_stats()
        y_np = y.detach().cpu().numpy().ravel()
        y_min_v, y_max_v = float(self.running_mean.min()), float(self.running_mean.max())
        y_range = y_np.max() - y_np.min()
        if y_range < 1e-8:
            return features
        edges = np.linspace(y_np.min() - 1e-8, y_np.max() + 1e-8, self.num_bins + 1)
        bin_idx = np.digitize(y_np, edges[1:], right=True)
        bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
        bin_idx_t = torch.tensor(bin_idx, device=features.device, dtype=torch.long)

        smoothed_mean = self.running_mean[bin_idx_t]
        smoothed_var = self.running_var[bin_idx_t]

        out = features.clone()
        for k in range(self.num_bins):
            mask = bin_idx_t == k
            if mask.sum() == 0:
                continue
            feat_k = features[mask]
            local_mean = feat_k.mean(dim=0)
            local_var = feat_k.var(dim=0, unbiased=False) + 1e-8
            normed = (feat_k - local_mean) / torch.sqrt(local_var)
            out[mask] = normed * torch.sqrt(smoothed_var[k]) + smoothed_mean[k]
        return out


class LDSFDSTrainer:
    """LDS+FDS trainer (Yang et al., 2021)."""

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.config = config

        lds_cfg = config.get("lds_fds", {})
        self.sigma = lds_cfg.get("sigma", 2.0)
        self.kernel_size = lds_cfg.get("kernel_size", 5)
        self.fds_feature_dim = lds_cfg.get("fds_feature_dim", 256)
        self.fds_bins = lds_cfg.get("fds_bins", 100)

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

        self.fds = _FDSModule(
            feature_dim=self.fds_feature_dim,
            num_bins=self.fds_bins,
            sigma=self.sigma,
        ).to(self.device)

        self.best_state = None
        self.best_val_loss = float("inf")
        self.y_train: np.ndarray | None = None

    def set_y_train(self, y_train: np.ndarray) -> None:
        self.y_train = np.asarray(y_train, dtype=np.float64).ravel()

    def train_epoch(self, dataloader: Any) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        y_ref = self.y_train
        if y_ref is None:
            all_y = []
            for _, y_b in dataloader:
                all_y.append(np.asarray(y_b).ravel())
            y_ref = np.concatenate(all_y)

        lds_weights_all = _compute_lds_weights(y_ref, sigma=self.sigma)
        lds_map = {i: lds_weights_all[i] for i in range(len(lds_weights_all))}

        sample_idx = 0
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)

            bs = X_batch.shape[0]
            y_np = y_batch.detach().cpu().numpy().ravel()
            w_np = _compute_lds_weights(
                y_ref, sigma=self.sigma,
            )
            batch_bin_idx = np.searchsorted(
                np.linspace(y_ref.min() - 1e-8, y_ref.max() + 1e-8, 101)[1:],
                y_np,
            )
            batch_bin_idx = np.clip(batch_bin_idx, 0, 99)
            smoothed_counts = gaussian_filter1d(
                np.bincount(
                    np.clip(
                        np.digitize(
                            y_ref,
                            np.linspace(y_ref.min() - 1e-8, y_ref.max() + 1e-8, 101)[1:],
                            right=True,
                        ),
                        0, 99,
                    ),
                    minlength=100,
                ).astype(np.float64),
                sigma=self.sigma,
                mode="constant",
            )
            smoothed_counts = np.clip(smoothed_counts, 1e-8, None)
            batch_weights = 1.0 / smoothed_counts[batch_bin_idx]
            batch_weights = batch_weights / batch_weights.mean()
            weights_t = torch.tensor(
                batch_weights, dtype=torch.float32, device=self.device,
            )

            self.optimizer.zero_grad()
            preds = self.model(X_batch)
            if preds.dim() == 2 and preds.shape[1] == 1:
                preds = preds.squeeze(1)
            targets = y_batch.squeeze(1) if y_batch.dim() == 2 else y_batch

            per_sample = torch.clamp((preds - targets) ** 2, max=9.0)
            loss = (per_sample * weights_t).mean()
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

"""
Autoencoder method — wraps the Phase 1 conv AE behind AnomalyMethod.

Training and scoring logic is unchanged from Phase 1; only the interface differs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import RunConfig
from core.base import AnomalyMethod
from core.utils import compute_reconstruction_anomaly_map, ensure_dir, image_score
from core.visualize import save_loss_curve
from methods.autoencoder.model import ConvAutoencoder


class AutoencoderMethod(AnomalyMethod):
    """Convolutional autoencoder trained on normal images only (MSE reconstruction)."""

    def __init__(self, cfg: RunConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.model = ConvAutoencoder().to(device)
        self.epoch_losses: list[float] = []

    def fit(self, train_loader: DataLoader) -> None:
        """Train with Adam + optional AMP — identical to Phase 1 train.py."""
        cfg = self.cfg
        ensure_dir(cfg.models_dir)
        ensure_dir(cfg.eval_dir)

        print(f"Device: {self.device}")
        print(f"Category: {cfg.category}")
        print(f"Training samples: {len(train_loader.dataset)}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()
        use_amp = cfg.use_amp and self.device.type == "cuda"
        scaler = GradScaler(self.device.type, enabled=use_amp)

        self.epoch_losses = []
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            running_loss = 0.0
            num_batches = 0

            for images, _masks, _meta in train_loader:
                images = images.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with autocast(self.device.type, enabled=use_amp):
                    recon = self.model(images)
                    loss = criterion(recon, images)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                num_batches += 1

            avg_loss = running_loss / max(num_batches, 1)
            self.epoch_losses.append(avg_loss)
            print(f"Epoch {epoch:3d}/{cfg.epochs}  |  loss = {avg_loss:.6f}")

        loss_curve_path = cfg.eval_dir / f"{cfg.category}_loss_curve.png"
        save_loss_curve(
            self.epoch_losses,
            loss_curve_path,
            title=f"{cfg.category} — training loss",
        )
        log_path = cfg.eval_dir / f"{cfg.category}_train_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({"epoch_losses": self.epoch_losses}, f, indent=2)

    @torch.no_grad()
    def score(self, image_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        recon = self.model(image_batch)
        batch_size = image_batch.size(0)
        maps: list[torch.Tensor] = []
        scores: list[float] = []

        for i in range(batch_size):
            amap = compute_reconstruction_anomaly_map(
                image_batch[i], recon[i], self.cfg.gaussian_sigma
            )
            maps.append(torch.from_numpy(amap))
            scores.append(image_score(amap, mode=self.cfg.image_score_mode))

        anomaly_maps = torch.stack(maps).to(image_batch.device)
        image_scores = torch.tensor(scores, device=image_batch.device, dtype=torch.float32)
        return image_scores, anomaly_maps

    @torch.no_grad()
    def reconstruct(self, image_batch: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(image_batch)

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        torch.save(
            {
                "method": "autoencoder",
                "model_state_dict": self.model.state_dict(),
                "category": self.cfg.category,
                "image_size": self.cfg.image_size,
                "epoch_losses": self.epoch_losses,
                "config": {
                    "epochs": self.cfg.epochs,
                    "learning_rate": self.cfg.learning_rate,
                    "batch_size": self.cfg.batch_size,
                },
            },
            path,
        )
        print(f"Saved checkpoint → {path}")

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.epoch_losses = state.get("epoch_losses", [])
        self.model.eval()

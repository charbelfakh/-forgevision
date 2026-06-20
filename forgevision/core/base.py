"""
Shared interface for all ForgeVision anomaly-detection methods.

Every method learns a model of "normal" from train/good/ images, then scores
test images with per-pixel anomaly maps and image-level scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader


class AnomalyMethod(ABC):
    """
    Abstract base class for swappable anomaly detectors.

    Lifecycle:
        1. fit(train_loader)  — learn from normal training images only
        2. score(batch)       — produce scores + heatmaps at inference
        3. save / load        — persist learned state to disk
    """

    @abstractmethod
    def fit(self, train_loader: DataLoader) -> None:
        """Learn the normal data distribution from training images."""

    @abstractmethod
    def score(
        self, image_batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score a batch of images.

        Args:
            image_batch: (B, 3, H, W) float tensor in [0, 1]

        Returns:
            image_scores:  (B,) — higher = more anomalous
            anomaly_maps:  (B, H, W) — per-pixel scores at image_size
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist method state (weights, memory bank, etc.)."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore method state from disk."""

    def reconstruct(self, image_batch: torch.Tensor) -> torch.Tensor | None:
        """
        Optional reconstruction for visualisation panels.

        Autoencoder methods override this; PatchCore returns None.
        """
        return None

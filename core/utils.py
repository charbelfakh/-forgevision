"""
Shared helpers: device selection, reproducibility, scoring utilities.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU. Override with FORGEVISION_DEVICE=cpu."""
    import os

    forced = os.environ.get("FORGEVISION_DEVICE", "").strip().lower()
    if forced == "cpu":
        return torch.device("cpu")
    if forced == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    """Fix random seeds so runs are reproducible (within hardware limits)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if missing; return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def smooth_anomaly_map(error_map: np.ndarray, gaussian_sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to a 2-D anomaly map (0 = no smoothing)."""
    if gaussian_sigma > 0:
        return gaussian_filter(error_map, sigma=gaussian_sigma).astype(np.float32)
    return error_map.astype(np.float32)


def image_score(anomaly_map: np.ndarray, mode: str = "mean") -> float:
    """
    Collapse a 2-D anomaly map into a single image-level score.

    mode="mean" — average (autoencoder default)
    mode="max"  — peak score (PatchCore default)
    """
    if mode == "mean":
        return float(anomaly_map.mean())
    if mode == "max":
        return float(anomaly_map.max())
    raise ValueError(f"Unknown image_score mode: {mode!r}. Use 'mean' or 'max'.")


def compute_reconstruction_anomaly_map(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    gaussian_sigma: float = 4.0,
) -> np.ndarray:
    """
    Per-pixel MSE reconstruction error map (autoencoder scoring).

    Args:
        original:       (C, H, W) in [0, 1]
        reconstruction: (C, H, W) in [0, 1]

    Returns:
        (H, W) float32 numpy array
    """
    diff = (original - reconstruction) ** 2
    error_map = diff.mean(dim=0).cpu().numpy()
    return smooth_anomaly_map(error_map, gaussian_sigma)

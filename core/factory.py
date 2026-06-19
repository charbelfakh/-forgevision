"""Factory for AnomalyMethod implementations."""

from __future__ import annotations

from config import RunConfig
from core.base import AnomalyMethod
from core.utils import get_device
from methods.autoencoder.method import AutoencoderMethod
from methods.patchcore.method import PatchCoreMethod


def create_method(cfg: RunConfig) -> AnomalyMethod:
    """Instantiate the requested anomaly-detection method."""
    device = get_device()
    if cfg.method == "autoencoder":
        return AutoencoderMethod(cfg, device)
    if cfg.method == "patchcore":
        return PatchCoreMethod(cfg, device)
    raise ValueError(f"Unknown method: {cfg.method!r}. Use 'autoencoder' or 'patchcore'.")

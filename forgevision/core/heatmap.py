"""
Verdict-based anomaly heatmap coloring.

The decision threshold (from eval/thresholds.json) sets the PASS/DEFECT badge only.
Colormap scaling is per-image percentile mapping so categories with different
absolute score scales (carpet vs pill) share the same visual language.

  vmin = low percentile of this image's patch scores → background blue/green
  vmax = high percentile → only the hottest region reaches red
"""

from __future__ import annotations

import math

import matplotlib.cm as cm
import numpy as np

# Per-image colormap anchors (typical/background vs localized peak).
PERCENTILE_VMIN = 60.0
PERCENTILE_VMAX = 99.0


def _valid_threshold(threshold: float | None) -> bool:
    if threshold is None:
        return False
    try:
        return math.isfinite(float(threshold))
    except (TypeError, ValueError):
        return False


def _normalize_percentile(anomaly_map: np.ndarray) -> np.ndarray:
    """
    Map patch scores to [0, 1] using this image's score distribution.

    Typical/background pixels sit near or below vmin (blue/green); only values
    at or above vmax (e.g. a defect spike) saturate to red.
    """
    amap = anomaly_map.astype(np.float32)
    vmin = float(np.percentile(amap, PERCENTILE_VMIN))
    vmax = float(np.percentile(amap, PERCENTILE_VMAX))
    if vmax - vmin < 1e-8:
        return np.zeros_like(amap, dtype=np.float32)
    norm = (amap - vmin) / (vmax - vmin)
    return np.clip(norm, 0.0, 1.0)


def anomaly_map_to_rgb(
    anomaly_map: np.ndarray,
    image_score: float,
    threshold: float | None,
) -> tuple[np.ndarray, str]:
    """
    Render anomaly map as jet RGB using per-image percentile scaling.

    Returns:
        (H, W, 3) uint8 RGB, mode label ("defect" | "normal" | "defect_fallback")
    """
    norm = _normalize_percentile(anomaly_map)

    if _valid_threshold(threshold) and float(image_score) >= float(threshold):
        mode = "defect"
    elif _valid_threshold(threshold):
        mode = "normal"
    else:
        mode = "defect_fallback"

    colored = cm.jet(norm)[..., :3]
    return (colored * 255).astype(np.uint8), mode


def blend_overlay(
    input_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend heatmap over the input image."""
    blended = (1.0 - alpha) * input_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)

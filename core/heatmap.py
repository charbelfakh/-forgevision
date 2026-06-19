"""
Verdict-based anomaly heatmap coloring.

The decision threshold (from eval/thresholds.json) picks the colormap branch only.
Scoring and PASS/DEFECT badges are unchanged elsewhere.

  DEFECT (score >= threshold): per-image min/max → jet (cool field, vivid red peak)
  NORMAL  (score <  threshold): scale into low jet range → calm deep blue
  Missing threshold: fall back to DEFECT-style per-image scaling (never crashes)
"""

from __future__ import annotations

import math

import matplotlib.cm as cm
import numpy as np

# PASS maps: norm capped here → jet blue end only (0 = deep blue, ~0.25 = cyan).
NORMAL_JET_MAX = 0.18


def _valid_threshold(threshold: float | None) -> bool:
    if threshold is None:
        return False
    try:
        return math.isfinite(float(threshold))
    except (TypeError, ValueError):
        return False


def _normalize_per_image(anomaly_map: np.ndarray) -> np.ndarray:
    """DEFECT branch: full dynamic range for high-contrast localization."""
    amap = anomaly_map.astype(np.float32)
    lo, hi = float(amap.min()), float(amap.max())
    if hi - lo < 1e-8:
        return np.zeros_like(amap, dtype=np.float32)
    return (amap - lo) / (hi - lo)


def _normalize_cool(anomaly_map: np.ndarray) -> np.ndarray:
    """NORMAL branch: entire map stays in the blue end of jet."""
    amap = anomaly_map.astype(np.float32)
    lo, hi = float(amap.min()), float(amap.max())
    span = max(hi - lo, 1e-8)
    norm = (amap - lo) / span * NORMAL_JET_MAX
    return np.clip(norm, 0.0, NORMAL_JET_MAX)


def anomaly_map_to_rgb(
    anomaly_map: np.ndarray,
    image_score: float,
    threshold: float | None,
) -> tuple[np.ndarray, str]:
    """
    Render anomaly map as jet RGB using verdict-based coloring.

    Returns:
        (H, W, 3) uint8 RGB, mode label ("defect" | "normal" | "defect_fallback")
    """
    if _valid_threshold(threshold) and float(image_score) >= float(threshold):
        norm = _normalize_per_image(anomaly_map)
        mode = "defect"
    elif _valid_threshold(threshold):
        norm = _normalize_cool(anomaly_map)
        mode = "normal"
    else:
        norm = _normalize_per_image(anomaly_map)
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

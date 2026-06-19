"""
Per-(category, method) decision thresholds from normal training scores.

Threshold = mean(normal_scores) + k * std(normal_scores)  (k=3 by default).

Heatmap coloring uses this threshold only to pick PASS vs DEFECT colormap branches
(see core/heatmap.py) — no separate vmin/vmax calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA_ROOT, EVAL_DIR, IMAGE_SIZE
from core.base import AnomalyMethod
from core.dataset import EvalTransform, MVTecDataset, build_sample_index

THRESHOLDS_PATH = EVAL_DIR / "thresholds.json"
THRESHOLD_STD_K = 3.0
FALLBACK_THRESHOLD = 0.5


def _load_store() -> dict:
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_store(data: dict) -> None:
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _key(category: str, method: str) -> str:
    return f"{category}::{method}"


def _calibrate_on_normals(
    category: str,
    method_instance: AnomalyMethod,
    data_root: Path,
    image_size: int,
) -> list[float]:
    """Score train/good/ images; return image-level scores for threshold fitting."""
    category_root = data_root / category
    train_samples, _ = build_sample_index(category_root)
    train_good = [s for s in train_samples if s.defect_type == "good"]
    if not train_good:
        return []

    ds = MVTecDataset(train_good, EvalTransform(image_size), image_size)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    scores: list[float] = []
    device = getattr(method_instance, "device", torch.device("cpu"))

    with torch.no_grad():
        for images, _masks, _meta in loader:
            batch_scores, _maps = method_instance.score(images.to(device))
            scores.extend(batch_scores.cpu().tolist())

    return scores


def get_calibration(
    category: str,
    method: str,
    method_instance: AnomalyMethod,
    *,
    data_root: Path | None = None,
    image_size: int = IMAGE_SIZE,
) -> dict:
    """
    Return calibration dict: threshold, is_default.

    Never raises on missing or partial thresholds.json entries.
    """
    store = _load_store()
    entry_key = _key(category, method)
    root = data_root or DATA_ROOT

    if entry_key in store and "threshold" in store[entry_key]:
        entry = store[entry_key]
        try:
            threshold = float(entry["threshold"])
        except (TypeError, ValueError):
            threshold = FALLBACK_THRESHOLD
        return {
            "threshold": threshold,
            "is_default": bool(entry.get("is_default", False)),
        }

    normal_scores = _calibrate_on_normals(category, method_instance, root, image_size)

    if len(normal_scores) >= 2:
        mean = float(np.mean(normal_scores))
        std = float(np.std(normal_scores))
        threshold = mean + THRESHOLD_STD_K * std
        is_default = False
    else:
        threshold = FALLBACK_THRESHOLD
        is_default = True
        mean = None
        std = None

    store[entry_key] = {
        "threshold": threshold,
        "is_default": is_default,
        "k_std": THRESHOLD_STD_K if not is_default else None,
        "n_normal_samples": len(normal_scores),
        "normal_score_mean": mean,
        "normal_score_std": std,
    }
    _save_store(store)

    return {"threshold": threshold, "is_default": is_default}


def get_threshold(
    category: str,
    method: str,
    method_instance: AnomalyMethod,
    *,
    data_root: Path | None = None,
    image_size: int = IMAGE_SIZE,
) -> tuple[float, bool]:
    """Backward-compatible wrapper."""
    cal = get_calibration(category, method, method_instance, data_root=data_root, image_size=image_size)
    return cal["threshold"], cal["is_default"]

"""
AUROC metrics for image-level and pixel-level anomaly detection evaluation.
"""

from __future__ import annotations

from sklearn.metrics import roc_auc_score


def compute_image_auroc(labels: list[int], scores: list[float]) -> float:
    """Image-level AUROC: good (0) vs defective (1). Returns NaN if only one class."""
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(labels: list[int], scores: list[float]) -> float:
    """Pixel-level AUROC: pooled over all test pixels. Returns NaN if no defect pixels."""
    if not labels or max(labels) == min(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def build_metrics_dict(
    category: str,
    image_auroc: float,
    pixel_auroc: float,
    image_labels: list[int],
    *,
    method: str = "",
    image_score_mode: str = "",
    gaussian_sigma: float = 0.0,
) -> dict:
    """Standard metrics payload returned by evaluate()."""
    return {
        "category": category,
        "method": method,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "image_score_mode": image_score_mode,
        "gaussian_sigma": gaussian_sigma,
        "num_test_images": len(image_labels),
        "num_good": sum(1 for label in image_labels if label == 0),
        "num_defective": sum(1 for label in image_labels if label == 1),
    }

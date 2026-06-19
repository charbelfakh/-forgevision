"""
Image preprocessing and heatmap rendering for the API layer.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from config import IMAGE_SIZE, VALID_METHODS
from core.heatmap import anomaly_map_to_rgb, blend_overlay
from core.utils import get_device


def validate_method(method: str) -> None:
    if method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method {method!r}. Choose from: {', '.join(VALID_METHODS)}"
        )


def load_image_tensor(image_bytes: bytes, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Decode uploaded bytes → (1, 3, H, W) float tensor in [0, 1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = TF.resize(img, [image_size, image_size])
    tensor = TF.to_tensor(img).unsqueeze(0)
    return tensor


def tensor_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) or (1, 3, H, W) → (H, W, 3) uint8."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def png_bytes_to_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")


def rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def render_heatmaps(
    input_tensor: torch.Tensor,
    anomaly_map: np.ndarray,
    *,
    image_score: float,
    threshold: float | None,
) -> tuple[str, str, str]:
    """
    Build base64 PNG strings for jet heatmap and alpha overlay.

    Colormap branch is chosen from score vs threshold (see core/heatmap.py).
    """
    input_rgb = tensor_to_uint8_rgb(input_tensor)
    heatmap_rgb, color_mode = anomaly_map_to_rgb(anomaly_map, image_score, threshold)
    overlay_rgb = blend_overlay(input_rgb, heatmap_rgb)

    heatmap_b64 = png_bytes_to_base64(rgb_to_png_bytes(heatmap_rgb))
    overlay_b64 = png_bytes_to_base64(rgb_to_png_bytes(overlay_rgb))
    return heatmap_b64, overlay_b64, color_mode


def run_inference(
    method_instance: Any,
    image_tensor: torch.Tensor,
    threshold: float,
    *,
    category: str,
    method: str,
) -> dict:
    """Score one image via AnomalyMethod.score() and package API response fields."""
    device = getattr(method_instance, "device", get_device())
    image_tensor = image_tensor.to(device)

    import time

    t0 = time.perf_counter()
    with torch.no_grad():
        scores, maps = method_instance.score(image_tensor)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    score = float(scores[0].cpu())
    amap = maps[0].cpu().numpy()
    heatmap_b64, overlay_b64, color_mode = render_heatmaps(
        image_tensor,
        amap,
        image_score=score,
        threshold=threshold,
    )

    return {
        "image_score": score,
        "is_anomaly": score > threshold,
        "threshold": threshold,
        "heatmap_color_mode": color_mode,
        "heatmap_png_base64": heatmap_b64,
        "overlay_png_base64": overlay_b64,
        "method": method,
        "category": category,
        "inference_ms": round(inference_ms, 2),
    }

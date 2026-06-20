"""
Visualisation helpers: example panels and training loss curves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

from forgevision.core.utils import ensure_dir


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float tensor in [0,1] to (H, W, 3) uint8 for plotting."""
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) mask tensor to (H, W) float32 in {0, 1}."""
    return mask.squeeze(0).cpu().numpy().astype(np.float32)


def save_loss_curve(
    losses: list[float],
    output_path: Path,
    title: str = "Training loss (MSE reconstruction)",
) -> None:
    """Save a simple epoch-vs-loss line plot."""
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(losses) + 1)
    ax.plot(epochs, losses, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_example_panel(
    original: torch.Tensor,
    reconstruction: torch.Tensor | None,
    anomaly_map: np.ndarray,
    mask: torch.Tensor,
    output_path: Path,
    image_label: str = "",
    image_score_value: float | None = None,
    recon_title: str = "Reconstruction",
) -> None:
    """
    Save a 4-panel figure: input | reconstruction | error heatmap | GT mask.

    If reconstruction is None (PatchCore), the second panel shows the input
    again with a "N/A" title.
    """
    ensure_dir(output_path.parent)
    orig_np = tensor_to_numpy_image(original)
    mask_np = mask_to_numpy(mask)

    if reconstruction is not None:
        recon_np = tensor_to_numpy_image(reconstruction)
    else:
        recon_np = orig_np
        recon_title = "N/A (no reconstruction)"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(orig_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(recon_np)
    axes[1].set_title(recon_title)
    axes[1].axis("off")

    im = axes[2].imshow(anomaly_map, cmap="jet")
    score_str = f" (score={image_score_value:.4f})" if image_score_value is not None else ""
    axes[2].set_title(f"Anomaly map{score_str}")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Ground-truth mask")
    axes[3].axis("off")

    if image_label:
        fig.suptitle(image_label, fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

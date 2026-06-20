"""
Method-agnostic evaluation loop using the AnomalyMethod interface.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from forgevision.config import RunConfig
from forgevision.core.base import AnomalyMethod
from forgevision.core.dataset import make_dataloaders
from forgevision.core.metrics import build_metrics_dict, compute_image_auroc, compute_pixel_auroc
from forgevision.core.utils import ensure_dir, get_device, set_seed
from forgevision.core.visualize import save_example_panel


def evaluate(cfg: RunConfig, method: AnomalyMethod) -> dict:
    """
    Run evaluation on the test set via the AnomalyMethod interface.

    Returns metrics dict and writes example panels + metrics.json.
    """
    set_seed(cfg.seed)
    device = get_device()
    ensure_dir(cfg.eval_dir)

    print(f"Device: {device}")
    print(f"Method: {cfg.method}")
    print(f"Category: {cfg.category}")

    _, test_loader = make_dataloaders(
        data_root=cfg.data_root,
        category=cfg.category,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    image_scores: list[float] = []
    image_labels: list[int] = []
    all_pixel_scores: list[float] = []
    all_pixel_labels: list[int] = []
    example_candidates: list[dict] = []

    with torch.no_grad():
        for images, masks, metas in test_loader:
            images = images.to(device, non_blocking=True)

            batch_scores, batch_maps = method.score(images)
            recons = method.reconstruct(images)

            for i in range(images.size(0)):
                label = int(metas["label"][i])
                defect_type = metas["defect_type"][i]
                image_path = metas["image_path"][i]

                anomaly_map = batch_maps[i].cpu().numpy()
                score = float(batch_scores[i].cpu())

                image_scores.append(score)
                image_labels.append(label)

                mask_np = masks[i].squeeze(0).cpu().numpy().astype(int)
                all_pixel_scores.extend(anomaly_map.ravel().tolist())
                all_pixel_labels.extend(mask_np.ravel().tolist())

                recon_i = recons[i].cpu() if recons is not None else None
                example_candidates.append(
                    {
                        "original": images[i].cpu(),
                        "reconstruction": recon_i,
                        "anomaly_map": anomaly_map,
                        "mask": masks[i].cpu(),
                        "label": label,
                        "defect_type": defect_type,
                        "image_path": image_path,
                        "score": score,
                    }
                )

    image_auroc = compute_image_auroc(image_labels, image_scores)
    pixel_auroc = compute_pixel_auroc(all_pixel_labels, all_pixel_scores)

    metrics = build_metrics_dict(
        cfg.category,
        image_auroc,
        pixel_auroc,
        image_labels,
        method=cfg.method,
        image_score_mode=cfg.image_score_mode,
        gaussian_sigma=cfg.gaussian_sigma,
    )

    metrics_path = cfg.eval_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nResults:")
    print(f"  Image AUROC: {image_auroc:.4f}")
    print(f"  Pixel AUROC: {pixel_auroc:.4f}")
    print(f"Saved metrics → {metrics_path}")

    defective = [e for e in example_candidates if e["label"] == 1]
    good = [e for e in example_candidates if e["label"] == 0]
    selected = defective[: cfg.num_example_panels]
    if len(selected) < cfg.num_example_panels:
        selected.extend(good[: cfg.num_example_panels - len(selected)])

    panels_dir = cfg.eval_dir / "examples" / cfg.method / cfg.category
    ensure_dir(panels_dir)

    for idx, ex in enumerate(selected):
        stem = Path(ex["image_path"]).stem
        label_str = "DEFECT" if ex["label"] == 1 else "GOOD"
        panel_path = panels_dir / f"{idx:02d}_{ex['defect_type']}_{stem}.png"
        save_example_panel(
            original=ex["original"],
            reconstruction=ex["reconstruction"],
            anomaly_map=ex["anomaly_map"],
            mask=ex["mask"],
            output_path=panel_path,
            image_label=f"{label_str} | {ex['defect_type']} | {stem}",
            image_score_value=ex["score"],
        )
        print(f"Saved example panel → {panel_path}")

    return metrics

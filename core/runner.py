"""
Orchestration: fit + evaluate one category via the AnomalyMethod interface.
"""

from __future__ import annotations

from pathlib import Path

from config import RunConfig, checkpoint_path
from core.base import AnomalyMethod
from core.dataset import make_dataloaders
from core.evaluate import evaluate
from core.factory import create_method


def run_category(cfg: RunConfig, *, retrain: bool) -> dict:
    """
    Train (if needed) and evaluate one category.

    Returns result row with image/pixel AUROC.
    """
    ckpt = cfg.checkpoint or checkpoint_path(cfg.category, cfg.method, cfg.models_dir)
    method = create_method(cfg)

    train_loader, test_loader = make_dataloaders(
        data_root=cfg.data_root,
        category=cfg.category,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment_train=(cfg.method == "autoencoder"),
    )

    if retrain or not ckpt.exists():
        print(f"Training {cfg.method} on {cfg.category} …")
        method.fit(train_loader)
        method.save(ckpt)
    else:
        print(f"  Skipping training — checkpoint exists: {ckpt}")
        method.load(ckpt)

    metrics = evaluate(cfg, method)
    return {
        "category": cfg.category,
        "image_auroc": metrics["image_auroc"],
        "pixel_auroc": metrics["pixel_auroc"],
        "error": "",
    }


def train_only(cfg: RunConfig) -> Path:
    """Train one category and return checkpoint path."""
    ckpt = cfg.checkpoint or checkpoint_path(cfg.category, cfg.method, cfg.models_dir)
    method = create_method(cfg)
    train_loader, _ = make_dataloaders(
        data_root=cfg.data_root,
        category=cfg.category,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment_train=(cfg.method == "autoencoder"),
    )
    method.fit(train_loader)
    method.save(ckpt)
    return ckpt


def eval_only(cfg: RunConfig) -> dict:
    """Evaluate one category from an existing checkpoint."""
    ckpt = cfg.checkpoint or checkpoint_path(cfg.category, cfg.method, cfg.models_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\nRun training first: python scripts/run_train.py"
        )
    method = create_method(cfg)
    method.load(ckpt)
    return evaluate(cfg, method)

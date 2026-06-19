#!/usr/bin/env python
"""
Train an anomaly-detection method on one MVTec category.

Usage:
    python scripts/run_train.py --category bottle --epochs 50
    python scripts/run_train.py --method patchcore --category bottle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    BATCH_SIZE,
    CORESET_RATIO,
    DATA_ROOT,
    DEFAULT_CATEGORY,
    DEFAULT_METHOD,
    EPOCHS,
    EVAL_DIR,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    NUM_WORKERS,
    SEED,
    USE_AMP,
    VALID_METHODS,
    RunConfig,
)
from core.runner import train_only  # noqa: E402
from core.utils import set_seed  # noqa: E402


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Train an anomaly-detection method.")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=VALID_METHODS)
    parser.add_argument("--category", type=str, default=DEFAULT_CATEGORY)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AE only)")
    parser.add_argument("--coreset-ratio", type=float, default=CORESET_RATIO)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()

    return RunConfig(
        method=args.method,
        category=args.category,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_amp=not args.no_amp and USE_AMP,
        num_workers=args.num_workers,
        seed=args.seed,
        coreset_ratio=args.coreset_ratio,
        models_dir=args.models_dir,
        eval_dir=args.eval_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    set_seed(cfg.seed)
    train_only(cfg)

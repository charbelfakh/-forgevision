#!/usr/bin/env python
"""
Evaluate a trained anomaly-detection method on one MVTec category.

Usage:
    python scripts/run_eval.py --category bottle
    python scripts/run_eval.py --method patchcore --category carpet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forgevision.config import (  # noqa: E402
    BATCH_SIZE,
    DATA_ROOT,
    DEFAULT_CATEGORY,
    DEFAULT_METHOD,
    EVAL_DIR,
    GAUSSIAN_SIGMA,
    IMAGE_SCORE_MODE,
    IMAGE_SIZE,
    MODELS_DIR,
    NUM_EXAMPLE_PANELS,
    NUM_WORKERS,
    SEED,
    VALID_METHODS,
    RunConfig,
)
from forgevision.core.runner import eval_only  # noqa: E402
from forgevision.core.utils import set_seed  # noqa: E402


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Evaluate an anomaly-detection method.")
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=VALID_METHODS)
    parser.add_argument("--category", type=str, default=DEFAULT_CATEGORY)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--image-score",
        type=str,
        default=IMAGE_SCORE_MODE,
        choices=["mean", "max"],
        help="Image score mode (autoencoder only; PatchCore always uses max)",
    )
    parser.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA)
    parser.add_argument("--num-examples", type=int, default=NUM_EXAMPLE_PANELS)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    args = parser.parse_args()

    return RunConfig(
        method=args.method,
        category=args.category,
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        models_dir=args.models_dir,
        eval_dir=args.eval_dir,
        image_score_mode=args.image_score,
        gaussian_sigma=args.gaussian_sigma,
        num_example_panels=args.num_examples,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    cfg = parse_args()
    set_seed(cfg.seed)
    eval_only(cfg)

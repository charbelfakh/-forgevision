#!/usr/bin/env python
"""
Train + evaluate an anomaly-detection method on all present MVTec categories.

Usage:
    python scripts/run_all.py --epochs 50
    python scripts/run_all.py --method patchcore
    python scripts/run_all.py --method autoencoder --categories bottle --retrain
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from config import (  # noqa: E402
    BATCH_SIZE,
    CORESET_RATIO,
    DATA_ROOT,
    DEFAULT_METHOD,
    EPOCHS,
    EVAL_DIR,
    MODELS_DIR,
    NUM_WORKERS,
    RunConfig,
    SEED,
    VALID_METHODS,
)
from core.aggregate import CANONICAL_CATEGORIES, aggregate_and_save, resolve_categories  # noqa: E402
from core.runner import run_category  # noqa: E402
from core.utils import set_seed  # noqa: E402


def _format_auroc(value: float) -> str:
    return f"{value:.2f}" if value == value else "NaN"


def _oom_message(category: str, cfg: RunConfig) -> str:
    if cfg.method == "patchcore":
        new_ratio = max(0.001, cfg.coreset_ratio / 2)
        return (
            f"CUDA OOM on '{category}'. "
            f"Try --batch-size {max(1, cfg.batch_size // 2)} or "
            f"--coreset-ratio {new_ratio:.4f} or "
            f"--categories {category} to retry alone."
        )
    return (
        f"CUDA OOM on '{category}'. "
        f"Re-run with --batch-size {max(1, cfg.batch_size // 2)} or "
        f"--categories {category} to retry this category alone."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train + evaluate an anomaly method on all present MVTec categories."
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD, choices=VALID_METHODS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--coreset-ratio", type=float, default=CORESET_RATIO)
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated subset (default: all present under data/mvtec_ad/)",
    )
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    subset = None
    if args.categories:
        subset = [c.strip() for c in args.categories.split(",") if c.strip()]

    to_run, present, missing = resolve_categories(args.data_root, subset)

    print("=" * 60)
    print(f"ForgeVision — multi-category run (method={args.method})")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Found {len(present)} categories with train/good/: {present}")
    if missing:
        print(f"Missing {len(missing)} of {len(CANONICAL_CATEGORIES)} canonical: {missing}")
    print(f"Will process {len(to_run)} categories: {to_run}")
    if args.method == "autoencoder":
        print(f"Epochs={args.epochs}  batch_size={args.batch_size}  retrain={args.retrain}")
    else:
        print(
            f"coreset_ratio={args.coreset_ratio}  batch_size={args.batch_size}  "
            f"retrain={args.retrain}"
        )
    print()

    if not to_run:
        print("Nothing to do — no categories found. Check data/mvtec_ad/.")
        sys.exit(1)

    results: list[dict] = []
    total = len(to_run)

    for idx, category in enumerate(to_run, start=1):
        print("-" * 60)
        print(f"[{idx}/{total}] {category}")

        cfg = RunConfig(
            method=args.method,
            category=category,
            data_root=args.data_root,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_workers=NUM_WORKERS,
            seed=args.seed,
            coreset_ratio=args.coreset_ratio,
            models_dir=args.models_dir,
            eval_dir=args.eval_dir,
        )

        try:
            row = run_category(cfg, retrain=args.retrain)
            results.append(row)
            print(
                f"[{idx}/{total}] {category}  "
                f"img={_format_auroc(row['image_auroc'])}  "
                f"px={_format_auroc(row['pixel_auroc'])}"
            )

        except torch.cuda.OutOfMemoryError:
            msg = _oom_message(category, cfg)
            print(f"ERROR: {msg}")
            results.append(
                {
                    "category": category,
                    "image_auroc": float("nan"),
                    "pixel_auroc": float("nan"),
                    "error": msg,
                }
            )

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"ERROR on '{category}': {msg}")
            traceback.print_exc()
            results.append(
                {
                    "category": category,
                    "image_auroc": float("nan"),
                    "pixel_auroc": float("nan"),
                    "error": msg,
                }
            )

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("Aggregating results …")
    aggregate_and_save(results, args.eval_dir, args.method)
    if args.method == "autoencoder":
        print(f"  → {args.eval_dir / 'results.md'}")
        print(f"  → {args.eval_dir / 'results.csv'}")
        print(f"  → {args.eval_dir / 'results_plot.png'}")
    else:
        print(f"  → {args.eval_dir / 'results_patchcore.csv'}")
        print(f"  → {args.eval_dir / 'comparison.md'}")
        print(f"  → {args.eval_dir / 'comparison_plot.png'}")
    print("Done.")


if __name__ == "__main__":
    main()

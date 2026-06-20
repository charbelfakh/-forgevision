#!/usr/bin/env python
"""
Precompute per-(category, method) thresholds from train/good scores.

Uses the same calibration logic as the API (api/thresholds.compute_calibration_entry).

Usage:
    python scripts/compute_thresholds.py
    python scripts/compute_thresholds.py --overwrite
    python scripts/compute_thresholds.py --categories bottle,carpet --methods patchcore
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.checkpoints import resolve_checkpoint  # noqa: E402
from api.thresholds import (  # noqa: E402
    THRESHOLDS_PATH,
    _load_store,
    _save_store,
    compute_calibration_entry,
)
from forgevision.config import DATA_ROOT, IMAGE_SIZE, MODELS_DIR, VALID_METHODS, RunConfig  # noqa: E402
from forgevision.core.aggregate import discover_present_categories  # noqa: E402
from forgevision.core.factory import create_method  # noqa: E402


def _store_key(category: str, method: str) -> str:
    return f"{category}::{method}"


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [part.strip() for part in value.split(",") if part.strip()]
    return items or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute eval/thresholds.json from train/good scores."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"MVTec AD root (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help=f"Checkpoint directory (default: {MODELS_DIR})",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories (default: all with train/good/)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(VALID_METHODS),
        help=f"Comma-separated methods (default: {','.join(VALID_METHODS)})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute entries already present in thresholds.json",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help=f"Input resolution (default: {IMAGE_SIZE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = _parse_csv_list(args.methods) or list(VALID_METHODS)
    unknown_methods = [m for m in methods if m not in VALID_METHODS]
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {unknown_methods}. Choose from: {list(VALID_METHODS)}")

    present = discover_present_categories(args.data_root)
    subset = _parse_csv_list(args.categories)
    if subset:
        unknown = [c for c in subset if c not in present]
        if unknown:
            raise SystemExit(
                f"Requested categories not found under {args.data_root}: {unknown}\n"
                f"Present with train/good/: {present}"
            )
        categories = subset
    else:
        categories = present

    store = _load_store()
    written: list[str] = []
    skipped: list[tuple[str, str]] = []

    print(f"Data root: {args.data_root}")
    print(f"Models dir: {args.models_dir}")
    print(f"Categories: {len(categories)} · Methods: {methods}")
    print(f"Output: {THRESHOLDS_PATH}")
    print()

    for category in categories:
        train_good = args.data_root / category / "train" / "good"
        if not train_good.is_dir():
            for method in methods:
                skipped.append((_store_key(category, method), "missing train/good"))
            continue

        for method in methods:
            key = _store_key(category, method)

            if key in store and "threshold" in store[key] and not args.overwrite:
                skipped.append((key, "already present"))
                continue

            ckpt = resolve_checkpoint(category, method, args.models_dir)
            if ckpt is None:
                skipped.append((key, "missing weights"))
                continue

            cfg = RunConfig(
                category=category,
                method=method,
                data_root=args.data_root,
                models_dir=args.models_dir,
                image_size=args.image_size,
            )
            model = create_method(cfg)
            model.load(ckpt)

            entry = compute_calibration_entry(
                category,
                model,
                data_root=args.data_root,
                image_size=args.image_size,
            )
            store[key] = entry
            written.append(key)
            print(
                f"  {key}: threshold={entry['threshold']:.6f} "
                f"(n={entry['n_normal_samples']}, default={entry['is_default']})"
            )

    _save_store(store)

    print()
    print(f"Written: {len(written)}")
    if written:
        for key in written:
            print(f"  + {key}")

    print(f"Skipped: {len(skipped)}")
    if skipped:
        by_reason: dict[str, list[str]] = {}
        for key, reason in skipped:
            by_reason.setdefault(reason, []).append(key)
        for reason, keys in sorted(by_reason.items()):
            print(f"  [{reason}] ({len(keys)})")
            for key in keys:
                print(f"    - {key}")

    print(f"\nStore now has {len(store)} entries at {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()

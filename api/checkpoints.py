"""
Discover trained model checkpoints on disk.

Supports both naming conventions:
  {category}_autoencoder.pth  (Phase 1–3 default)
  {category}_ae.pth           (shorthand)
  {category}_patchcore.pth
"""

from __future__ import annotations

from pathlib import Path

from config import MODELS_DIR, VALID_METHODS


def resolve_checkpoint(
    category: str,
    method: str,
    models_dir: Path | None = None,
) -> Path | None:
    """Return the first existing checkpoint path for (category, method), or None."""
    root = models_dir or MODELS_DIR
    if method == "autoencoder":
        candidates = [
            root / f"{category}_autoencoder.pth",
            root / f"{category}_ae.pth",
        ]
    elif method == "patchcore":
        candidates = [root / f"{category}_patchcore.pth"]
    else:
        return None

    for path in candidates:
        if path.is_file():
            return path
    return None


def discover_categories(models_dir: Path | None = None) -> list[dict]:
    """
    Scan models/ and return categories with available methods.

    Returns:
        [{"category": "bottle", "methods": ["autoencoder", "patchcore"]}, ...]
    """
    root = models_dir or MODELS_DIR
    if not root.is_dir():
        return []

    catalog: dict[str, set[str]] = {}

    for path in root.glob("*.pth"):
        stem = path.stem
        if stem.endswith("_patchcore"):
            cat = stem[: -len("_patchcore")]
            catalog.setdefault(cat, set()).add("patchcore")
        elif stem.endswith("_autoencoder"):
            cat = stem[: -len("_autoencoder")]
            catalog.setdefault(cat, set()).add("autoencoder")
        elif stem.endswith("_ae"):
            cat = stem[: -len("_ae")]
            catalog.setdefault(cat, set()).add("autoencoder")

    return [
        {
            "category": cat,
            "methods": sorted(m for m in catalog[cat] if m in VALID_METHODS),
        }
        for cat in sorted(catalog)
        if catalog[cat]
    ]

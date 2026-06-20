"""
Lazy-load and cache AnomalyMethod instances keyed by (category, method).
"""

from __future__ import annotations

import threading
from pathlib import Path

from forgevision.config import MODELS_DIR, RunConfig
from forgevision.core.base import AnomalyMethod
from forgevision.core.factory import create_method

from api.checkpoints import resolve_checkpoint

_cache: dict[tuple[str, str], AnomalyMethod] = {}
_lock = threading.Lock()


def get_method(
    category: str,
    method: str,
    models_dir: Path | None = None,
) -> AnomalyMethod:
    """
    Return a loaded model, fetching from cache or disk on first use.

    Raises:
        FileNotFoundError: no checkpoint for this (category, method)
    """
    key = (category, method)
    if key in _cache:
        return _cache[key]

    with _lock:
        if key in _cache:
            return _cache[key]

        ckpt = resolve_checkpoint(category, method, models_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f"No weights found for category={category!r}, method={method!r}. "
                f"Expected under {models_dir or MODELS_DIR}"
            )

        cfg = RunConfig(category=category, method=method, models_dir=models_dir or MODELS_DIR)
        instance = create_method(cfg)
        instance.load(ckpt)
        _cache[key] = instance
        return instance


def clear_cache() -> None:
    """Drop all cached models (useful for tests or VRAM cleanup)."""
    with _lock:
        _cache.clear()

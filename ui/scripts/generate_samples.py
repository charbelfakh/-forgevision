"""Generate synthetic sample PNGs for the UI (not MVTec — safe to commit)."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "public" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def make_normal(size: int = 256) -> Image.Image:
    rng = np.random.default_rng(42)
    base = rng.integers(80, 120, (size, size, 3), dtype=np.uint8)
    img = Image.fromarray(base, mode="RGB")
    return img


def make_defect(size: int = 256) -> Image.Image:
    img = make_normal(size)
    draw = ImageDraw.Draw(img)
    draw.ellipse([90, 90, 170, 170], fill=(220, 40, 40))
    return img


if __name__ == "__main__":
    make_normal().save(OUT / "normal_pattern.png")
    make_defect().save(OUT / "defect_pattern.png")
    print(f"Wrote samples to {OUT}")

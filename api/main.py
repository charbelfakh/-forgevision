"""
ForgeVision inference API — thin serving layer over core/ + methods/.

Start with:
    uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

# Project root on sys.path so `api` and repo-relative paths resolve when not installed.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.checkpoints import discover_categories, resolve_checkpoint
from api.inference import load_image_tensor, run_inference, validate_method
from api.model_cache import get_method
from api.thresholds import get_calibration
from forgevision.config import MODELS_DIR

import os

app = FastAPI(
    title="ForgeVision API",
    description="Industrial visual anomaly detection — autoencoder & PatchCore",
    version="0.5.0",
)

_cors = os.environ.get(
    "FORGEVISION_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080,http://127.0.0.1:8080",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/categories")
def categories() -> dict:
    """
    Categories with at least one trained method on disk.

    Auto-category detection is out of scope — the client must pick a category.
    """
    items = discover_categories(MODELS_DIR)
    return {"categories": items}


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="RGB image (PNG/JPEG)"),
    category: str = Form(..., description="MVTec category, e.g. bottle"),
    method: str = Form(..., description="autoencoder or patchcore"),
) -> dict:
    """
    Run anomaly detection on an uploaded image.

    Requires trained weights at models/{category}_{method}.pth (or _ae.pth for AE).
    """
    category = category.strip().lower()
    method = method.strip().lower()

    if not category:
        raise HTTPException(status_code=400, detail="category is required")
    try:
        validate_method(method)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if resolve_checkpoint(category, method, MODELS_DIR) is None:
        available = discover_categories(MODELS_DIR)
        raise HTTPException(
            status_code=404,
            detail=(
                f"No trained weights for category={category!r}, method={method!r}. "
                f"Available: {available}"
            ),
        )

    try:
        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty image file")

        image_tensor = load_image_tensor(raw)
        method_instance = get_method(category, method, MODELS_DIR)
        cal = get_calibration(category, method, method_instance)

        result = run_inference(
            method_instance,
            image_tensor,
            cal["threshold"],
            category=category,
            method=method,
        )
        result["threshold_is_default"] = cal["is_default"]
        return result

    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc

"""
Central configuration for ForgeVision anomaly detection.

All paths are relative to the project root (the folder containing this file).
Scripts override these via CLI flags — see scripts/run_*.py.
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DEFAULT_CATEGORY = "bottle"
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec_ad"
IMAGE_SIZE = 256
NUM_WORKERS = 0

# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------
DEFAULT_METHOD = "autoencoder"
VALID_METHODS = ("autoencoder", "patchcore")

# ---------------------------------------------------------------------------
# Training (autoencoder)
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3
USE_AMP = True

MODELS_DIR = PROJECT_ROOT / "models"
EVAL_DIR = PROJECT_ROOT / "eval"


def checkpoint_path(category: str, method: str = DEFAULT_METHOD, models_dir: Path | None = None) -> Path:
    """Standard checkpoint filename: models/{category}_{method}.pth"""
    root = models_dir or MODELS_DIR
    suffix = "autoencoder" if method == "autoencoder" else method
    return root / f"{category}_{suffix}.pth"


# ---------------------------------------------------------------------------
# Evaluation — autoencoder
# ---------------------------------------------------------------------------
IMAGE_SCORE_MODE = "mean"
GAUSSIAN_SIGMA = 4.0
NUM_EXAMPLE_PANELS = 4
SEED = 42

# ---------------------------------------------------------------------------
# PatchCore-specific
# ---------------------------------------------------------------------------
CORESET_RATIO = 0.01          # ~1% of training patches — key 8 GB VRAM lever
NN_QUERY_CHUNK = 2048         # query patches per NN distance batch
NN_MEMORY_CHUNK = 4096        # memory rows per NN distance batch


@dataclass
class RunConfig:
    """Unified runtime config for train, eval, and orchestration."""

    method: str = DEFAULT_METHOD
    category: str = DEFAULT_CATEGORY
    data_root: Path = field(default_factory=lambda: DATA_ROOT)
    image_size: int = IMAGE_SIZE
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    seed: int = SEED
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)
    eval_dir: Path = field(default_factory=lambda: EVAL_DIR)
    checkpoint: Path | None = None

    # Autoencoder
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    use_amp: bool = USE_AMP
    image_score_mode: str = IMAGE_SCORE_MODE
    gaussian_sigma: float = GAUSSIAN_SIGMA
    num_example_panels: int = NUM_EXAMPLE_PANELS

    # PatchCore
    coreset_ratio: float = CORESET_RATIO
    nn_query_chunk: int = NN_QUERY_CHUNK
    nn_memory_chunk: int = NN_MEMORY_CHUNK


# Backward-compatible aliases used by older code paths
TrainConfig = RunConfig
EvalConfig = RunConfig

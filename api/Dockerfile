# ForgeVision backend — CPU PyTorch by default (GPU optional via README).
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORGEVISION_DEVICE=cpu \
    PIP_NO_CACHE_DIR=1

# System deps for Pillow / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (lean image; swap for CUDA build for GPU — see README)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY api/requirements-docker.txt /app/api/requirements-docker.txt
RUN pip install -r /app/api/requirements-docker.txt

# Application code + trained weights (models/ must exist in build context)
COPY config.py /app/config.py
COPY core/ /app/core/
COPY methods/ /app/methods/
COPY api/ /app/api/

RUN mkdir -p /app/eval
COPY eval/thresholds.json /app/eval/thresholds.json

COPY models/ /app/models/

RUN test -n "$(find /app/models -name '*.pth' -print -quit)" || \
    (echo "ERROR: No .pth weights in models/ — train models before docker build" && exit 1)

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

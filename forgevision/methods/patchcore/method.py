"""
PatchCore anomaly detection (Roth et al., 2022).

Approach (from scratch, no anomalib):
  1. Extract frozen WideResNet-50 patch embeddings from normal training images.
  2. Build a memory bank via greedy coreset subsampling (~1% of patches).
  3. Score test patches by nearest-neighbour distance to the memory bank.
  4. Upsample patch scores → image_size, Gaussian smooth, max-pool for image score.

Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", 2022.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from forgevision.config import RunConfig
from forgevision.core.base import AnomalyMethod
from forgevision.core.utils import ensure_dir, smooth_anomaly_map
from forgevision.methods.patchcore.backbone import PatchCoreBackbone
from forgevision.methods.patchcore.coreset import greedy_coreset_indices
from forgevision.methods.patchcore.nn_search import nearest_neighbour_distances


class PatchCoreMethod(AnomalyMethod):
    """PatchCore: memory-bank anomaly detection with pretrained backbone features."""

    def __init__(self, cfg: RunConfig, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.backbone = PatchCoreBackbone().to(device)
        self.memory_bank: torch.Tensor | None = None
        self.spatial_size: tuple[int, int] = (16, 16)

    def _extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalised patch embeddings from a batch.

        Returns:
            (B * h * w, D) tensor on CPU (keeps GPU memory free for large banks).
        """
        images = PatchCoreBackbone.imagenet_normalize(images)
        with torch.no_grad():
            feat_map, self.spatial_size = self.backbone(images)
        b, c, h, w = feat_map.shape
        patches = feat_map.permute(0, 2, 3, 1).reshape(b * h * w, c)
        patches = F.normalize(patches, dim=1)
        return patches.cpu()

    def fit(self, train_loader: DataLoader) -> None:
        """Collect normal patch features and subsample into a coreset memory bank."""
        cfg = self.cfg
        print(f"Device: {self.device}")
        print(f"Category: {cfg.category}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print("Extracting patch features from normal training images …")

        all_patches: list[torch.Tensor] = []
        for images, _masks, _meta in train_loader:
            images = images.to(self.device, non_blocking=True)
            patches = self._extract_patch_features(images)
            all_patches.append(patches)

        features = torch.cat(all_patches, dim=0)
        n_total = features.shape[0]
        n_coreset = max(1, int(n_total * cfg.coreset_ratio))
        print(f"Total patches: {n_total:,}  →  coreset target: {n_coreset:,} ({cfg.coreset_ratio:.1%})")

        indices = greedy_coreset_indices(features, n_coreset, seed=cfg.seed)
        self.memory_bank = features[indices].contiguous()

        # Try keeping memory on GPU for faster NN search; fall back to CPU if tight.
        try:
            self.memory_bank = self.memory_bank.to(self.device)
            _ = nearest_neighbour_distances(
                self.memory_bank[: min(64, len(self.memory_bank))],
                self.memory_bank,
                query_chunk=64,
                memory_chunk=256,
                use_gpu=True,
            )
            print(f"Memory bank on {self.device} ({self.memory_bank.shape[0]} × {self.memory_bank.shape[1]})")
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.memory_bank = self.memory_bank.cpu()
            print(f"Memory bank on CPU ({self.memory_bank.shape[0]} × {self.memory_bank.shape[1]}) — GPU OOM during probe")

    @torch.no_grad()
    def score(self, image_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore memory bank not built — call fit() or load() first.")

        cfg = self.cfg
        self.backbone.eval()
        images = image_batch.to(self.device, non_blocking=True)
        b = images.size(0)
        h, w = self.spatial_size
        image_size = cfg.image_size

        batch_scores: list[torch.Tensor] = []
        batch_maps: list[torch.Tensor] = []

        for i in range(b):
            patches = self._extract_patch_features(images[i : i + 1])
            use_gpu = self.memory_bank.device.type == "cuda"
            try:
                dists = nearest_neighbour_distances(
                    patches,
                    self.memory_bank,
                    query_chunk=cfg.nn_query_chunk,
                    memory_chunk=cfg.nn_memory_chunk,
                    use_gpu=use_gpu,
                )
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                dists = nearest_neighbour_distances(
                    patches,
                    self.memory_bank.cpu(),
                    query_chunk=cfg.nn_query_chunk,
                    memory_chunk=cfg.nn_memory_chunk,
                    use_gpu=False,
                )

            patch_map = dists.reshape(h, w).to(self.device)
            upsampled = F.interpolate(
                patch_map.unsqueeze(0).unsqueeze(0),
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            amap_np = smooth_anomaly_map(upsampled.cpu().numpy(), cfg.gaussian_sigma)
            amap_tensor = torch.from_numpy(amap_np).float()

            # Standard PatchCore image score = max of the smoothed anomaly map.
            img_score = amap_tensor.max()
            batch_scores.append(img_score)
            batch_maps.append(amap_tensor)

        image_scores = torch.stack(batch_scores).to(image_batch.device)
        anomaly_maps = torch.stack(batch_maps).to(image_batch.device)
        return image_scores, anomaly_maps

    def save(self, path: Path) -> None:
        if self.memory_bank is None:
            raise RuntimeError("Nothing to save — memory bank is empty.")
        ensure_dir(path.parent)
        torch.save(
            {
                "method": "patchcore",
                "memory_bank": self.memory_bank.cpu(),
                "spatial_size": self.spatial_size,
                "category": self.cfg.category,
                "image_size": self.cfg.image_size,
                "coreset_ratio": self.cfg.coreset_ratio,
                "gaussian_sigma": self.cfg.gaussian_sigma,
            },
            path,
        )
        print(f"Saved checkpoint → {path}")

    def load(self, path: Path) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.memory_bank = state["memory_bank"]
        self.spatial_size = tuple(state["spatial_size"])
        try:
            self.memory_bank = self.memory_bank.to(self.device)
        except torch.cuda.OutOfMemoryError:
            print("Warning: could not move memory bank to GPU — using CPU for NN search.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

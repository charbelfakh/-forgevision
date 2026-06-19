"""
WideResNet-50 feature extractor for PatchCore.

Uses frozen ImageNet-pretrained layer2 + layer3 features, aligned spatially
and concatenated into per-patch embedding vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class PatchCoreBackbone(nn.Module):
    """
    Extract multi-scale patch features from a frozen WideResNet-50.

    For 256×256 input:
        layer2 → (B, 512, 32, 32)
        layer3 → (B, 1024, 16, 16)
    layer2 is adaptively average-pooled to 16×16, then concatenated with layer3
    → (B, 1536, 16, 16) patch feature map.
    """

    def __init__(self) -> None:
        super().__init__()
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for param in self.parameters():
            param.requires_grad = False

        self.eval()
        self.out_channels = 512 + 1024  # 1536

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised

        Returns:
            features: (B, C, h, w) patch feature map
            spatial_size: (h, w)
        """
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)

        target_hw = f3.shape[-2:]
        f2_aligned = F.adaptive_avg_pool2d(f2, target_hw)
        features = torch.cat([f2_aligned, f3], dim=1)
        return features, (target_hw[0], target_hw[1])

    @staticmethod
    def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
        """Convert [0,1] RGB tensors to ImageNet-normalised tensors."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

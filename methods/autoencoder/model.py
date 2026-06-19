"""
Fully convolutional autoencoder for unsupervised anomaly detection.

Architecture (256×256 RGB input):
    Encoder: 5 stride-2 conv blocks  → 256 → 128 → 64 → 32 → 16 → 8
    Bottleneck: 1×1 conv (keeps spatial resolution at 8×8)
    Decoder: 5 transposed-conv blocks → back to 256×256
    Output: Sigmoid → reconstruction in [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Encoder block: Conv2d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """Decoder block: ConvTranspose2d → BatchNorm → ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvAutoencoder(nn.Module):
    """Symmetric conv autoencoder for image reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 256)
        self.enc5 = ConvBlock(256, 256)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.dec5 = DeconvBlock(256, 256)
        self.dec4 = DeconvBlock(256, 256)
        self.dec3 = DeconvBlock(256, 128)
        self.dec2 = DeconvBlock(128, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return self.bottleneck(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec5(z)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        return self.dec1(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

"""
MVTec AD dataset loader.

MVTec AD layout (per category, e.g. bottle):
    train/good/              ← normal images used for training ONLY
    test/good/               ← normal test images (label = good)
    test/<defect_type>/      ← defective test images
    ground_truth/<defect>/   ← pixel masks named <image_stem>_mask.png

Training uses train/good/ only (unsupervised anomaly detection paradigm).
At test time we score every image and compare against labels / pixel masks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class SampleInfo:
    """One image path plus its split, label, and optional mask path."""

    image_path: Path
    split: str
    label: int
    defect_type: str
    mask_path: Path | None = None


class TrainTransform:
    """
    Resize → light augmentation → ToTensor.

    Augmentation is intentionally mild: MVTec defects are subtle, and heavy
    augmentation can make the autoencoder learn to reconstruct unrealistic
    views of "normal" rather than the true defect-free distribution.
    """

    def __init__(self, image_size: int = 256) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(image, [self.image_size, self.image_size])
        if random.random() < 0.5:
            image = TF.hflip(image)
        angle = random.uniform(-5.0, 5.0)
        image = TF.rotate(image, angle)
        return TF.to_tensor(image)


class EvalTransform:
    """Deterministic resize + ToTensor for validation / test."""

    def __init__(self, image_size: int = 256) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = TF.resize(image, [self.image_size, self.image_size])
        return TF.to_tensor(image)


def load_mask(mask_path: Path | None, image_size: int) -> torch.Tensor:
    """
    Load a pixel-level ground-truth mask.

    Returns (1, H, W) float tensor with values in {0, 1}.
    Good (normal) test images have no mask file → all-zero mask.
    """
    if mask_path is None or not mask_path.exists():
        return torch.zeros(1, image_size, image_size, dtype=torch.float32)

    mask = Image.open(mask_path).convert("L")
    mask = TF.resize(
        mask,
        [image_size, image_size],
        interpolation=transforms.InterpolationMode.NEAREST,
    )
    mask_tensor = TF.to_tensor(mask)
    return (mask_tensor > 0.5).float()


def _collect_images(folder: Path) -> list[Path]:
    """Return sorted list of image files in a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in extensions)


def build_sample_index(category_root: Path) -> tuple[list[SampleInfo], list[SampleInfo]]:
    """Walk the MVTec AD folder tree and build train / test sample lists."""
    train_dir = category_root / "train" / "good"
    train_samples = [
        SampleInfo(
            image_path=p,
            split="train",
            label=0,
            defect_type="good",
            mask_path=None,
        )
        for p in _collect_images(train_dir)
    ]

    test_samples: list[SampleInfo] = []

    for p in _collect_images(category_root / "test" / "good"):
        test_samples.append(
            SampleInfo(
                image_path=p,
                split="test",
                label=0,
                defect_type="good",
                mask_path=None,
            )
        )

    test_root = category_root / "test"
    if test_root.is_dir():
        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir() or defect_dir.name == "good":
                continue
            defect_type = defect_dir.name
            gt_dir = category_root / "ground_truth" / defect_type
            for p in _collect_images(defect_dir):
                mask_path = gt_dir / f"{p.stem}_mask.png"
                test_samples.append(
                    SampleInfo(
                        image_path=p,
                        split="test",
                        label=1,
                        defect_type=defect_type,
                        mask_path=mask_path if mask_path.exists() else None,
                    )
                )

    return train_samples, test_samples


class MVTecDataset(Dataset):
    """PyTorch Dataset for one MVTec AD category."""

    def __init__(
        self,
        samples: list[SampleInfo],
        transform: TrainTransform | EvalTransform,
        image_size: int = 256,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        info = self.samples[idx]
        image = Image.open(info.image_path).convert("RGB")
        image = self.transform(image)
        mask = load_mask(info.mask_path, self.image_size)

        meta = {
            "image_path": str(info.image_path),
            "mask_path": str(info.mask_path) if info.mask_path else "",
            "label": info.label,
            "defect_type": info.defect_type,
            "split": info.split,
        }
        return image, mask, meta


def make_dataloaders(
    data_root: Path,
    category: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augment_train: bool = True,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train and test DataLoaders for one MVTec category."""
    from torch.utils.data import DataLoader

    category_root = data_root / category
    if not category_root.is_dir():
        raise FileNotFoundError(
            f"Category folder not found: {category_root}\n"
            f"Download MVTec AD and unzip so the layout is:\n"
            f"  {data_root}/<category>/train/good/ ..."
        )

    train_samples, test_samples = build_sample_index(category_root)
    if not train_samples:
        raise RuntimeError(f"No training images found in {category_root / 'train' / 'good'}")
    if not test_samples:
        raise RuntimeError(f"No test images found under {category_root / 'test'}")

    train_transform: TrainTransform | EvalTransform = (
        TrainTransform(image_size) if augment_train else EvalTransform(image_size)
    )
    train_ds = MVTecDataset(train_samples, train_transform, image_size)
    test_ds = MVTecDataset(test_samples, EvalTransform(image_size), image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader

"""Dataset helpers for retinal vessel segmentation experiments."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def split_image_mask_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
    val_fraction: float = 0.2,
    seed: int = 41,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    masks_by_stem = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    pairs = [
        (image_path, masks_by_stem[image_path.stem])
        for image_path in image_dir.iterdir()
        if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.stem in masks_by_stem
    ]
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {image_dir} and {mask_dir}. Filenames must share stems.")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_count = max(1, int(round(len(pairs) * val_fraction))) if len(pairs) > 1 else 0
    return pairs[val_count:], pairs[:val_count]


class VesselSegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        image_size: int = 512,
        augment: bool = False,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        if self.augment and random.random() < 0.5:
            image_array = np.flip(image_array, axis=1).copy()
            mask_array = np.flip(mask_array, axis=1).copy()
        if self.augment and random.random() < 0.5:
            image_array = np.flip(image_array, axis=0).copy()
            mask_array = np.flip(mask_array, axis=0).copy()

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor

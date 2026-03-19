from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# Canonical class order — index == integer label fed to the model
CLASS_NAMES: list[str] = [
    "Real",              # 0  — genuine images
    "StyleGan2",         # 1  — seen fake
    "StyleGan3",         # 2  — seen fake
    "Glide",             # 3  — seen fake
    "Gated Convolution", # 4  — seen fake
    "Taming Transformer",# 5  — seen fake
    "Unseen",            # 6  — unseen fake (few samples in train, majority in test)
]

CLASS_TO_IDX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
)


class ArtiFactDataset(Dataset):
    """
    PyTorch Dataset for the ArtiFact fake-image-detection benchmark.

    Expected directory layout (standard ImageFolder structure):

        root/
          train/
            Real/
            StyleGan2/
            ...
            Unseen/          ← sparse in train
          val/
            ...
          test/
            Unseen/          ← majority of unseen samples live here

    Parameters
    ----------
    root : str | Path
        Path to the dataset root (contains train/, val/, test/).
    split : str
        One of "train", "val", "test".
    transform : callable, optional
        Albumentations Compose pipeline.  Receives a numpy HWC uint8 image,
        must return a dict with key "image" (CHW float tensor).
    class_names : list[str], optional
        Override the canonical CLASS_NAMES order / subset.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.class_names: list[str] = class_names if class_names is not None else CLASS_NAMES
        self.class_to_idx: dict[str, int] = {
            name: idx for idx, name in enumerate(self.class_names)
        }

        self.samples: list[tuple[Path, int]] = self._scan_samples()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_samples(self) -> list[tuple[Path, int]]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Expected layout: <root>/{self.split}/<class_name>/<image_files>"
            )

        samples: list[tuple[Path, int]] = []
        missing: list[str] = []

        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                missing.append(class_name)
                continue

            label = self.class_to_idx[class_name]
            for entry in sorted(class_dir.iterdir()):
                if entry.suffix.lower() in _IMAGE_EXTENSIONS:
                    samples.append((entry, label))

        if missing:
            print(
                f"[ArtiFactDataset] Warning: {self.split} split is missing class "
                f"folders: {missing}.  Continuing with available classes."
            )

        if not samples:
            raise RuntimeError(
                f"No images found under {split_dir}. "
                "Check that class folder names match CLASS_NAMES."
            )

        return samples

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except (UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Failed to load image {img_path}: {exc}") from exc

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_class_counts(self) -> list[int]:
        """Return per-class sample counts in dataset order."""
        counts = [0] * len(self.class_names)
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def __repr__(self) -> str:
        counts = self.get_class_counts()
        lines = [f"ArtiFactDataset(split={self.split!r}, total={len(self)})"]
        for name, count in zip(self.class_names, counts):
            lines.append(f"  {name:<20s} {count:>8,d}")
        return "\n".join(lines)

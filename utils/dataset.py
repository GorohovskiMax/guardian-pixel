from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Class definitions                                                            #
# --------------------------------------------------------------------------- #

# Human-readable class names — index matches the integer label in master_metadata.csv.
CLASS_NAMES: list[str] = [
    "Real",               # 0 — all genuine sources
    "StyleGan2",          # 1 — seen fake
    "StyleGan3",          # 2 — seen fake
    "Glide",              # 3 — seen fake
    "Gated Convolution",  # 4 — seen fake
    "Taming Transformer", # 5 — seen fake
    "Unseen",             # 6 — unseen fake generators (val/test only)
]

NUM_CLASSES = len(CLASS_NAMES)


# --------------------------------------------------------------------------- #
# Path resolution                                                              #
# --------------------------------------------------------------------------- #

def _build_prefix_map(artifact_root: Path) -> dict[str, str]:
    """
    Scan one level deep under artifact_root and return a mapping from
    subfolder name → parent folder name for any subfolder that is NOT itself
    a top-level folder.

    Example: artifact_root/glide/glide-t2i/ exists but glide-t2i/ does not
    exist at the top level, so the map contains {"glide-t2i": "glide"}.

    This lets us resolve CSV image_path values that are relative to a
    generator parent folder rather than to artifact_root directly.
    """
    top_level = {p.name for p in artifact_root.iterdir() if p.is_dir()}
    prefix_map: dict[str, str] = {}
    for folder in top_level:
        for sub in (artifact_root / folder).iterdir():
            if sub.is_dir() and sub.name not in top_level:
                prefix_map[sub.name] = folder
    return prefix_map


def _resolve_path(artifact_root: Path, image_path: str, prefix_map: dict[str, str]) -> str:
    """Return the absolute path for an image, inserting a parent folder when needed."""
    prefix = image_path.split("/")[0]
    parent = prefix_map.get(prefix, "")
    if parent:
        return str(artifact_root / parent / image_path)
    return str(artifact_root / image_path)


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

class ArtiFactDataset(Dataset):
    """
    PyTorch Dataset for the ArtiFact fake-image-detection benchmark.

    Data access
    -----------
    All metadata is read from a single master CSV file.  Images are never
    copied or reorganised — each image is fetched directly by constructing:

        artifact_root / image_path

    where ``image_path`` is the value stored in the master CSV.

    Parameters
    ----------
    csv_path : str | Path
        Path to master_metadata.csv (on Google Drive or local).
    artifact_root : str | Path
        Root directory where the ArtiFact zip was extracted, e.g.
        ``/content/ArtiFact``.  Every ``image_path`` in the CSV is
        relative to this root.
    split : str
        One of ``"train"``, ``"validation"``, ``"test"``.
    transform : callable, optional
        Albumentations ``Compose`` pipeline.  Receives a numpy HWC uint8
        image and must return ``{"image": CHW float tensor}``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        artifact_root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        assert split in ("train", "validation", "test"), f"Unknown split: {split!r}"

        self.artifact_root = Path(artifact_root)
        self.split         = split
        self.transform     = transform
        self.class_names   = CLASS_NAMES

        df = self._load_split(Path(csv_path), split)

        # Some generators are nested one level inside a parent folder on disk
        # (e.g. image_path starts with "glide-t2i/" but on disk the file lives
        # under "glide/glide-t2i/").  Build a one-time prefix map at startup
        # to resolve these transparently without touching the CSV.
        prefix_map = _build_prefix_map(self.artifact_root)

        # Each sample is a (absolute_image_path, class_label) tuple.
        self.samples: list[tuple[str, int]] = list(
            zip(
                (_resolve_path(self.artifact_root, p, prefix_map) for p in df["image_path"]),
                df["target"].tolist(),
            )
        )

    # ---------------------------------------------------------------------- #
    # Internal                                                                #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _load_split(csv_path: Path, split: str) -> pd.DataFrame:
        """Read the master CSV and return only rows belonging to ``split``."""
        if not csv_path.exists():
            raise FileNotFoundError(
                f"master_metadata.csv not found at: {csv_path}\n"
                "Make sure Google Drive is mounted and the path is correct."
            )
        df = pd.read_csv(csv_path, usecols=["image_path", "target", "split"], low_memory=False)
        subset = df[df["split"] == split].reset_index(drop=True)
        if len(subset) == 0:
            raise ValueError(
                f"No rows found for split={split!r} in {csv_path}.\n"
                f"Available splits: {df['split'].unique().tolist()}"
            )
        return subset

    # ---------------------------------------------------------------------- #
    # Dataset protocol                                                        #
    # ---------------------------------------------------------------------- #

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

    # ---------------------------------------------------------------------- #
    # Utility                                                                 #
    # ---------------------------------------------------------------------- #

    def get_class_counts(self) -> list[int]:
        """Return per-class sample counts aligned with CLASS_NAMES order."""
        counts = [0] * NUM_CLASSES
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def __repr__(self) -> str:
        counts = self.get_class_counts()
        lines  = [f"ArtiFactDataset(split={self.split!r}, total={len(self):,})"]
        for name, count in zip(CLASS_NAMES, counts):
            lines.append(f"  {name:<20s} {count:>10,d}")
        return "\n".join(lines)

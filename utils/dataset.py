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

# Module-level cache — the file index is built once per session and reused
# across all three ArtiFactDataset instances (train / validation / test).
_file_index_cache: dict[str, dict[str, list[str]]] = {}


# --------------------------------------------------------------------------- #
# Path resolution                                                              #
# --------------------------------------------------------------------------- #

def _build_file_index(artifact_root: Path) -> dict[str, list[str]]:
    """
    Walk the entire ArtiFact tree once and build a filename index.

    Returns
    -------
    dict mapping filename (e.g. "img001158.jpg") → list of relative paths
    from artifact_root where that file exists on disk.

    This handles arbitrary nesting depths transparently — no matter how
    many subdirectory levels a generator uses, its images will be found.
    The index is cached at module level so it is built only once per session.
    """
    print("[ArtiFactDataset] Building file index (one-time per session)...", flush=True)
    index: dict[str, list[str]] = {}
    for p in artifact_root.rglob("*.jpg"):
        rel = str(p.relative_to(artifact_root)).replace("\\", "/")
        fname = p.name
        if fname not in index:
            index[fname] = []
        index[fname].append(rel)
    total = sum(len(v) for v in index.values())
    print(f"[ArtiFactDataset] Index ready: {total:,} images.", flush=True)
    return index


def _get_file_index(artifact_root: Path) -> dict[str, list[str]]:
    """Return the cached file index, building it first if necessary."""
    key = str(artifact_root)
    if key not in _file_index_cache:
        _file_index_cache[key] = _build_file_index(artifact_root)
    return _file_index_cache[key]


def _resolve_path(artifact_root: Path, image_path: str, file_index: dict[str, list[str]]) -> str:
    """
    Resolve a CSV image_path to an absolute disk path.

    Strategy
    --------
    1. Extract the filename from image_path.
    2. Look it up in the file index (all known disk locations).
    3. If only one match exists, use it directly.
    4. If multiple matches exist (same filename in different generators),
       pick the candidate whose path shares the most components with the
       original CSV image_path — this reliably selects the correct generator.
    5. If no match exists, return the direct path so the caller gets a
       clear FileNotFoundError with the attempted path.
    """
    fname = Path(image_path).name
    candidates = file_index.get(fname, [])

    if not candidates:
        # File not found anywhere on disk — return direct path for a clear error.
        return str(artifact_root / image_path)

    if len(candidates) == 1:
        return str(artifact_root / candidates[0])

    # Multiple files share this filename — pick the one whose path components
    # overlap most with the original CSV image_path.
    original_parts = set(image_path.replace("\\", "/").split("/"))
    best = max(candidates, key=lambda c: len(original_parts & set(c.split("/"))))
    return str(artifact_root / best)


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

class ArtiFactDataset(Dataset):
    """
    PyTorch Dataset for the ArtiFact fake-image-detection benchmark.

    Data access
    -----------
    All metadata is read from a single master CSV file.  Images are fetched
    directly by resolving each ``image_path`` value against the disk.

    Path resolution is done via a full file index built by walking
    ``artifact_root`` once at startup.  This handles any nesting depth
    transparently — no generator-specific path knowledge is required.

    Parameters
    ----------
    csv_path : str | Path
        Path to master_metadata.csv (on Google Drive or local).
    artifact_root : str | Path
        Root directory where the ArtiFact zip was extracted, e.g.
        ``/content/ArtiFact``.
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

        df         = self._load_split(Path(csv_path), split)
        file_index = _get_file_index(self.artifact_root)

        # Each sample is a (absolute_image_path, class_label) tuple.
        self.samples: list[tuple[str, int]] = list(
            zip(
                (_resolve_path(self.artifact_root, p, file_index) for p in df["image_path"]),
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

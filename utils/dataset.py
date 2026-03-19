from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Class definitions                                                            #
# --------------------------------------------------------------------------- #

# Human-readable class names — index is the integer label fed to the model.
CLASS_NAMES: list[str] = [
    "Real",               # 0 — all genuine sources
    "StyleGan2",          # 1 — seen fake
    "StyleGan3",          # 2 — seen fake
    "Glide",              # 3 — seen fake
    "Gated Convolution",  # 4 — seen fake  (folder: generative_inpainting)
    "Taming Transformer", # 5 — seen fake  (folder: taming_transformer)
    "Unseen",             # 6 — unseen fake (all remaining fake generators)
]

# Generator folder name → 7-class integer label.
# Any fake folder NOT listed here is treated as Unseen (class 6).
# Real images are identified by target==0 in metadata, regardless of folder.
_SEEN_FAKE_TO_CLASS: dict[str, int] = {
    "stylegan2":             1,
    "stylegan3":             2,
    "glide":                 3,
    "generative_inpainting": 4,
    "taming_transformer":    5,
}

# --------------------------------------------------------------------------- #
# Split configuration                                                          #
# --------------------------------------------------------------------------- #

# Seen data (classes 0-5): stratified 70 / 15 / 15
_SEEN_TRAIN_RATIO = 0.70
_SEEN_VAL_RATIO   = 0.15
# test = 1 - 0.70 - 0.15 = 0.15

# Unseen fakes (class 6): excluded from train; 10% val, 90% test
_UNSEEN_VAL_RATIO = 0.10

_RANDOM_STATE = 42  # fixed seed — all three splits must see the same partitioning


# --------------------------------------------------------------------------- #
# Helper                                                                       #
# --------------------------------------------------------------------------- #

def _assign_class(generator: str, target: int) -> int:
    """Map a (generator folder, metadata target) pair to a 7-class label."""
    if target == 0:
        return 0  # Real — independent of which source folder it came from
    return _SEEN_FAKE_TO_CLASS.get(generator, 6)  # seen fake or Unseen


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

class ArtiFactDataset(Dataset):
    """
    PyTorch Dataset for the ArtiFact fake-image-detection benchmark.

    Dataset layout (actual on-disk structure)
    ------------------------------------------
    Each generator / real-source has its own top-level folder containing
    image files and a ``metadata.csv``:

        root/
          stylegan2/
            metadata.csv       ← columns: image_path, target, category
            <images …>
          glide/
            metadata.csv
            <images …>
          imagenet/
            metadata.csv
            <images …>
          …

    ``image_path`` values in each CSV are interpreted as relative to ``root``
    (the dataset root), not to the individual generator subfolder.

    Class mapping
    -------------
    - target == 0  →  class 0  (Real)
    - stylegan2    →  class 1  (StyleGan2)
    - stylegan3    →  class 2  (StyleGan3)
    - glide        →  class 3  (Glide)
    - generative_inpainting  →  class 4  (Gated Convolution)
    - taming_transformer     →  class 5  (Taming Transformer)
    - any other fake         →  class 6  (Unseen)

    Splits
    ------
    Built deterministically from the full metadata (random_state=42):
      - train : 70 % of seen data (classes 0-5); unseen fakes excluded
      - val   : 15 % of seen data + 10 % of unseen fakes
      - test  : 15 % of seen data + 90 % of unseen fakes

    Parameters
    ----------
    root : str | Path
        Path to the dataset root (contains generator subfolders).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    transform : callable, optional
        Albumentations ``Compose`` pipeline.  Receives a numpy HWC uint8
        image, must return ``{"image": CHW float tensor}``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"

        self.root      = Path(root)
        self.split     = split
        self.transform = transform
        self.class_names = CLASS_NAMES

        full_df  = self._aggregate_metadata()
        split_df = self._make_split(full_df)

        # samples: list of (abs_image_path: str, class_label: int)
        self.samples: list[tuple[str, int]] = list(
            zip(split_df["abs_path"].tolist(), split_df["class_label"].tolist())
        )

    # ---------------------------------------------------------------------- #
    # Metadata aggregation                                                    #
    # ---------------------------------------------------------------------- #

    def _aggregate_metadata(self) -> pd.DataFrame:
        """
        Walk every generator subfolder, read its metadata.csv, assign class
        labels, and return the combined DataFrame.
        """
        frames: list[pd.DataFrame] = []
        skipped: list[str] = []

        for gen_dir in sorted(self.root.iterdir()):
            if not gen_dir.is_dir():
                continue
            csv_path = gen_dir / "metadata.csv"
            if not csv_path.exists():
                skipped.append(gen_dir.name)
                continue

            df = pd.read_csv(csv_path, usecols=["image_path", "target"])
            df["generator"]   = gen_dir.name
            # image_path is relative to the dataset root
            df["abs_path"]    = df["image_path"].apply(
                lambda p: str(self.root / p)
            )
            df["class_label"] = df.apply(
                lambda row: _assign_class(gen_dir.name, int(row["target"])),
                axis=1,
            )
            frames.append(df[["abs_path", "generator", "class_label"]])

        if skipped:
            print(f"[ArtiFactDataset] Skipped (no metadata.csv): {skipped}")
        if not frames:
            raise RuntimeError(
                f"No metadata.csv files found under {self.root}.\n"
                "Expected: <root>/<generator>/metadata.csv"
            )

        return pd.concat(frames, ignore_index=True)

    # ---------------------------------------------------------------------- #
    # Split construction                                                      #
    # ---------------------------------------------------------------------- #

    def _make_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministically partition the full metadata DataFrame into
        train / val / test and return the slice for ``self.split``.
        """
        seen_df   = df[df["class_label"] < 6].reset_index(drop=True)
        unseen_df = df[df["class_label"] == 6].reset_index(drop=True)

        # ------------------------------------------------------------------ #
        # Seen data: stratified 70 / 15 / 15                                 #
        # ------------------------------------------------------------------ #

        # Step 1 — carve out 70 % for training
        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=round(1.0 - _SEEN_TRAIN_RATIO, 10),
            random_state=_RANDOM_STATE,
        )
        train_idx, rest_idx = next(sss1.split(seen_df, seen_df["class_label"]))
        train_df  = seen_df.iloc[train_idx]
        rest_seen = seen_df.iloc[rest_idx]

        # Step 2 — split the remaining 30 % evenly: 15 % val, 15 % test
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.5,
            random_state=_RANDOM_STATE,
        )
        val_idx, test_idx = next(sss2.split(rest_seen, rest_seen["class_label"]))
        val_seen_df  = rest_seen.iloc[val_idx]
        test_seen_df = rest_seen.iloc[test_idx]

        # ------------------------------------------------------------------ #
        # Unseen fakes: 0 % train, 10 % val, 90 % test                       #
        # ------------------------------------------------------------------ #

        if len(unseen_df) > 1:
            sss3 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=round(1.0 - _UNSEEN_VAL_RATIO, 10),
                random_state=_RANDOM_STATE,
            )
            # Stratify on generator so each unseen generator is proportionally
            # represented in val and test; fall back to class label if needed.
            strat_col = (
                unseen_df["generator"]
                if unseen_df["generator"].nunique() > 1
                else unseen_df["class_label"]
            )
            u_val_idx, u_test_idx = next(sss3.split(unseen_df, strat_col))
            val_unseen_df  = unseen_df.iloc[u_val_idx]
            test_unseen_df = unseen_df.iloc[u_test_idx]
        else:
            # Edge case: 0 or 1 unseen sample — all goes to test
            val_unseen_df  = pd.DataFrame(columns=df.columns)
            test_unseen_df = unseen_df

        # ------------------------------------------------------------------ #
        # Return the requested split                                          #
        # ------------------------------------------------------------------ #

        if self.split == "train":
            return train_df
        if self.split == "val":
            return pd.concat([val_seen_df, val_unseen_df], ignore_index=True)
        return pd.concat([test_seen_df, test_unseen_df], ignore_index=True)

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
        counts = [0] * len(CLASS_NAMES)
        for _, label in self.samples:
            counts[label] += 1
        return counts

    def __repr__(self) -> str:
        counts = self.get_class_counts()
        lines  = [f"ArtiFactDataset(split={self.split!r}, total={len(self):,})"]
        for name, count in zip(CLASS_NAMES, counts):
            lines.append(f"  {name:<20s} {count:>10,d}")
        return "\n".join(lines)

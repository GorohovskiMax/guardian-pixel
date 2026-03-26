from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import ArtiFactDataset
from .transforms import get_transforms


def get_dataloaders(
    config_path: str | Path,
    csv_path: Optional[str | Path] = None,
    artifact_root: Optional[str | Path] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> dict[str, DataLoader]:
    """
    Build train / validation / test DataLoaders for the ArtiFact dataset.

    Reads ``batch_size``, ``num_workers``, ``data.csv_path``,
    ``data.artifact_root``, ``model.input_resolution``, and
    ``training.samples_per_epoch`` from the YAML config.

    Returns
    -------
    dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping to DataLoaders.
    """
    cfg = _load_config(config_path)
    d_cfg = cfg["data"]

    csv_path          = Path(csv_path      if csv_path      is not None else d_cfg["csv_path"])
    artifact_root     = Path(artifact_root if artifact_root is not None else d_cfg["artifact_root"])
    batch_size        = batch_size  if batch_size  is not None else cfg["training"]["batch_size"]
    num_workers       = num_workers if num_workers is not None else d_cfg["num_workers"]
    input_res         = cfg["model"]["input_resolution"]
    samples_per_epoch = cfg["training"].get("samples_per_epoch", None)

    # ------------------------------------------------------------------ #
    # Datasets                                                             #
    # ------------------------------------------------------------------ #
    datasets = {
        split: ArtiFactDataset(
            csv_path=csv_path,
            artifact_root=artifact_root,
            split=split,
            transform=get_transforms(split, input_resolution=input_res),
        )
        for split in ("train", "validation", "test")
    }

    # ------------------------------------------------------------------ #
    # Weighted sampler — inverse class frequency on the training set       #
    # ------------------------------------------------------------------ #
    #
    # samples_per_epoch caps how many samples are drawn per epoch.
    # With replacement=True the sampler can draw the same image multiple
    # times, but the weighting ensures all 7 classes are equally represented
    # regardless of how many images each class actually has.
    #
    train_sampler = _make_weighted_sampler(datasets["train"], samples_per_epoch)

    # ------------------------------------------------------------------ #
    # DataLoaders                                                          #
    # ------------------------------------------------------------------ #
    pin = torch.cuda.is_available()

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        ),
    }

    return loaders


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f)


def _make_weighted_sampler(
    dataset: ArtiFactDataset,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """
    Compute per-sample weights as inverse class frequency and return a
    WeightedRandomSampler.

    Parameters
    ----------
    num_samples : int, optional
        Number of samples to draw per epoch.  Defaults to the full dataset
        size.  Set this to a smaller value (e.g. 500_000) to cap epoch
        length when the training set is very large.
    """
    counts = dataset.get_class_counts()
    class_weights = [
        1.0 / count if count > 0 else 0.0
        for count in counts
    ]
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples if num_samples is not None else len(sample_weights),
        replacement=True,
    )

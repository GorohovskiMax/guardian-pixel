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
    ``data.artifact_root``, and ``model.input_resolution`` from the YAML
    config.  ``csv_path``, ``artifact_root``, ``batch_size``, and
    ``num_workers`` may be overridden at call time.

    Parameters
    ----------
    config_path : str | Path
        Path to a layer config YAML (e.g. ``configs/layer_a.yaml``).
    csv_path : str | Path, optional
        Path to master_metadata.csv — overrides ``data.csv_path`` in config.
    artifact_root : str | Path, optional
        ArtiFact extraction root — overrides ``data.artifact_root`` in config.
    batch_size : int, optional
        Override config ``training.batch_size``.
    num_workers : int, optional
        Override config ``data.num_workers``.

    Returns
    -------
    dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping to DataLoaders.
    """
    cfg = _load_config(config_path)
    d_cfg = cfg["data"]

    csv_path      = Path(csv_path      if csv_path      is not None else d_cfg["csv_path"])
    artifact_root = Path(artifact_root if artifact_root is not None else d_cfg["artifact_root"])
    batch_size    = batch_size  if batch_size  is not None else cfg["training"]["batch_size"]
    num_workers   = num_workers if num_workers is not None else d_cfg["num_workers"]
    input_res     = cfg["model"]["input_resolution"]

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
    # The training set is imbalanced across the 7 classes.
    # WeightedRandomSampler rebalances this so each class contributes
    # roughly equally to every epoch.
    #
    train_sampler = _make_weighted_sampler(datasets["train"])

    # ------------------------------------------------------------------ #
    # DataLoaders                                                          #
    # ------------------------------------------------------------------ #
    pin = torch.cuda.is_available()

    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=train_sampler,   # mutually exclusive with shuffle=True
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
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


def _make_weighted_sampler(dataset: ArtiFactDataset) -> WeightedRandomSampler:
    """
    Compute per-sample weights as the inverse frequency of their class,
    then return a WeightedRandomSampler over the full training set.
    """
    counts = dataset.get_class_counts()
    class_weights = [
        1.0 / count if count > 0 else 0.0
        for count in counts
    ]
    sample_weights = [class_weights[label] for _, label in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from layers.forensic import ForensicDetector
from utils.dataloader import get_dataloaders


# --------------------------------------------------------------------------- #
# train_one_epoch                                                              #
# --------------------------------------------------------------------------- #

def _set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: ForensicDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    use_wandb: bool = True,
) -> tuple[dict, int]:
    """
    Run one full pass over the training set.

    The scheduler is stepped once per batch (not per epoch) to honour the
    step-based ``decay_steps`` parameter in the config.

    Parameters
    ----------
    global_step : int
        Running batch counter passed in from the outer loop so that W&B
        step numbers are consistent across epochs.

    Returns
    -------
    metrics : dict
        ``loss`` and ``accuracy`` averaged over the epoch.
    global_step : int
        Updated counter after this epoch's batches.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3} [train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()   # step-level decay

        running_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        global_step += 1

        if use_wandb:
            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": running_loss / len(loader),
        "accuracy": correct / total,
    }, global_step


# --------------------------------------------------------------------------- #
# evaluate                                                                     #
# --------------------------------------------------------------------------- #

def evaluate(
    model: ForensicDetector,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate the model and return loss, accuracy, and balanced accuracy.

    Balanced accuracy is the macro-average of per-class recall, computed via
    ``sklearn.metrics.balanced_accuracy_score``.  It is the primary metric
    for checkpoint selection because the Unseen class is severely imbalanced
    in the val split.

    Returns
    -------
    dict with keys: ``loss``, ``accuracy``, ``balanced_accuracy``.
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()  # no label smoothing for eval
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            running_loss += criterion(logits, labels).item()

            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss":              running_loss / len(loader),
        "accuracy":          sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
    }


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #

def train(
    config_path: str | Path,
    data_root: Optional[str | Path] = None,
    wandb_project: str = "guardian-pixel",
    wandb_run_name: Optional[str] = None,
) -> ForensicDetector:
    """
    Full training run for ForensicDetector.

    Reads all hyperparameters from ``config_path``.  ``data_root`` and
    ``wandb_run_name`` may be overridden at call time (useful in Colab where
    the Drive path differs from the default config value).

    Checkpoint strategy
    -------------------
    A single ``best.pt`` file is kept under ``training.checkpoint_dir``.
    It is overwritten whenever val balanced accuracy improves, so the file
    always holds the best weights seen so far.

    Parameters
    ----------
    config_path : str | Path
        Path to a layer config YAML (e.g. ``configs/layer_a.yaml``).
    data_root : str | Path, optional
        Override ``data.root`` from the config (e.g. a Google Drive path).
    wandb_project : str
        W&B project name.
    wandb_run_name : str, optional
        Human-readable run name for W&B.

    Returns
    -------
    ForensicDetector
        The model in its final (last-epoch) state.
        Load ``best.pt`` separately if you need the best checkpoint.
    """
    config_path = Path(config_path)
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    t_cfg = cfg["training"]
    s_cfg = t_cfg["scheduler"]

    # ------------------------------------------------------------------ #
    # Seeds                                                                #
    # ------------------------------------------------------------------ #
    seed = t_cfg.get("seed", 42)
    _set_seeds(seed)
    print(f"[train] random seed: {seed}")

    # ------------------------------------------------------------------ #
    # Device                                                               #
    # ------------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}")

    # ------------------------------------------------------------------ #
    # W&B                                                                  #
    # ------------------------------------------------------------------ #
    run_name = wandb_run_name or cfg.get("logging", {}).get("wandb_run_name")
    use_wandb = True
    try:
        wandb.init(project=wandb_project, name=run_name, config=cfg)
    except Exception as exc:
        use_wandb = False
        print(f"[train] W&B unavailable ({exc}) — training without logging.")

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    loaders = get_dataloaders(config_path, root=data_root)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = ForensicDetector.from_config(config_path).to(device)
    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=200)

    # ------------------------------------------------------------------ #
    # Loss                                                                 #
    # ------------------------------------------------------------------ #
    criterion = nn.CrossEntropyLoss(label_smoothing=t_cfg["label_smoothing"])

    # ------------------------------------------------------------------ #
    # Optimizer                                                            #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["learning_rate"])

    # ------------------------------------------------------------------ #
    # Scheduler — continuous exponential decay, applied per batch         #
    #                                                                      #
    # Formula: lr = lr_0 * decay_rate ^ (step / decay_steps)             #
    # This matches the TF/Keras ExponentialDecay semantics used in the    #
    # ArtiFact paper.  At step=0 the multiplier is exactly 1.0.           #
    # ------------------------------------------------------------------ #
    decay_rate  = s_cfg["decay_rate"]
    decay_steps = s_cfg["decay_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: decay_rate ** (step / decay_steps),
    )

    # ------------------------------------------------------------------ #
    # Checkpoint                                                           #
    # ------------------------------------------------------------------ #
    checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    best_bal_acc = 0.0

    # ------------------------------------------------------------------ #
    # Loop                                                                 #
    # ------------------------------------------------------------------ #
    global_step = 0

    for epoch in range(1, t_cfg["epochs"] + 1):

        train_metrics, global_step = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            device, scheduler, epoch, global_step, use_wandb=use_wandb,
        )

        val_metrics = evaluate(model, loaders["val"], device)

        if use_wandb:
            wandb.log(
                {
                    "epoch":                  epoch,
                    "train/epoch_loss":       train_metrics["loss"],
                    "train/epoch_accuracy":   train_metrics["accuracy"],
                    "val/loss":               val_metrics["loss"],
                    "val/accuracy":           val_metrics["accuracy"],
                    "val/balanced_accuracy":  val_metrics["balanced_accuracy"],
                },
                step=global_step,
            )

        print(
            f"Epoch {epoch:>3}/{t_cfg['epochs']}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best checkpoint
        if val_metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = val_metrics["balanced_accuracy"]
            torch.save(
                {
                    "epoch":                    epoch,
                    "model_state_dict":         model.state_dict(),
                    "optimizer_state_dict":     optimizer.state_dict(),
                    "scheduler_state_dict":     scheduler.state_dict(),
                    "best_balanced_accuracy":   best_bal_acc,
                    "config":                   cfg,
                    "rng_state":                torch.get_rng_state(),
                    "cuda_rng_state":           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                best_checkpoint_path,
            )
            if use_wandb:
                wandb.log(
                    {"val/best_balanced_accuracy": best_bal_acc},
                    step=global_step,
                )
            print(f"  → checkpoint saved  (bal_acc={best_bal_acc:.4f})")

    if use_wandb:
        wandb.finish()
    return model

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from layers.forensic import ForensicDetector
from utils.dataloader import get_dataloaders
from utils.dataset import CLASS_NAMES


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #

def _set_seeds(seed: int) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------- #
# train_one_epoch                                                              #
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: ForensicDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    use_wandb: bool = True,
) -> tuple[dict, int]:
    """
    Run one full pass over the training set with automatic mixed precision (AMP).

    AMP runs the forward pass in float16, cutting VRAM usage by ~2x and
    speeding up GPU kernels on modern hardware.  The GradScaler handles
    loss scaling to keep float16 gradients numerically stable.

    The scheduler is stepped once per batch (not per epoch) to honour the
    step-based ``decay_steps`` parameter in the config.
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
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

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
        "loss":     running_loss / len(loader),
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
    Evaluate the model and return a comprehensive set of metrics.

    7-class metrics
    ---------------
    - loss                  Cross-entropy loss averaged over batches
    - accuracy              Overall accuracy across all 7 classes
    - balanced_accuracy     Macro-average per-class recall (primary metric for
                            checkpoint selection — robust to class imbalance)
    - per_class_report      Per-class precision, recall, F1
    - confusion_matrix      7×7 numpy array

    Binary metrics (Real vs. Fake)
    --------------------------------
    Any prediction with label > 0 is collapsed to "fake" (1); label == 0
    stays "real" (0).  This mirrors the external binary output of the system.

    - binary_accuracy       Binary classification accuracy
    - binary_precision      Precision for the fake class
    - binary_recall         Recall for the fake class (= detection rate)
    - binary_f1             F1 score for the fake class
    - binary_roc_auc        Area under the ROC curve
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds:  list[int]   = []
    all_labels: list[int]   = []
    all_probs:  list[float] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)

            running_loss += criterion(logits, labels).item()

            probs = torch.softmax(logits.float(), dim=1)
            all_probs.extend((1.0 - probs[:, 0]).cpu().tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # ------------------------------------------------------------------ #
    # 7-class metrics                                                      #
    # ------------------------------------------------------------------ #
    accuracy     = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    bal_accuracy = balanced_accuracy_score(all_labels, all_preds)
    cm           = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    present_classes = sorted(set(all_labels))
    present_names   = [CLASS_NAMES[i] for i in present_classes]
    per_class = classification_report(
        all_labels, all_preds,
        labels=present_classes,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )

    # ------------------------------------------------------------------ #
    # Binary collapse: 0 = Real, 1 = Fake                                 #
    # ------------------------------------------------------------------ #
    binary_labels = [0 if l == 0 else 1 for l in all_labels]
    binary_preds  = [0 if p == 0 else 1 for p in all_preds]

    bin_tp = sum(p == 1 and l == 1 for p, l in zip(binary_preds, binary_labels))
    bin_fp = sum(p == 1 and l == 0 for p, l in zip(binary_preds, binary_labels))
    bin_fn = sum(p == 0 and l == 1 for p, l in zip(binary_preds, binary_labels))
    bin_tn = sum(p == 0 and l == 0 for p, l in zip(binary_preds, binary_labels))

    binary_accuracy  = (bin_tp + bin_tn) / len(binary_labels)
    binary_precision = bin_tp / (bin_tp + bin_fp) if (bin_tp + bin_fp) > 0 else 0.0
    binary_recall    = bin_tp / (bin_tp + bin_fn) if (bin_tp + bin_fn) > 0 else 0.0
    binary_f1        = (
        2 * binary_precision * binary_recall / (binary_precision + binary_recall)
        if (binary_precision + binary_recall) > 0 else 0.0
    )

    try:
        binary_roc_auc = roc_auc_score(binary_labels, all_probs)
    except ValueError:
        binary_roc_auc = float("nan")

    return {
        "loss":              running_loss / len(loader),
        "accuracy":          accuracy,
        "balanced_accuracy": bal_accuracy,
        "per_class_report":  per_class,
        "confusion_matrix":  cm,
        "binary_accuracy":   binary_accuracy,
        "binary_precision":  binary_precision,
        "binary_recall":     binary_recall,
        "binary_f1":         binary_f1,
        "binary_roc_auc":    binary_roc_auc,
    }


# --------------------------------------------------------------------------- #
# train                                                                        #
# --------------------------------------------------------------------------- #

def train(
    config_path: str | Path,
    csv_path: str | Path,
    artifact_root: str | Path,
    wandb_project: str = "guardian-pixel",
    wandb_run_name: str = None,
) -> ForensicDetector:
    """
    Full training run for ForensicDetector.

    Training uses the ``train`` split from master_metadata.csv.
    After each epoch, the model is evaluated on the ``validation`` split.
    The best checkpoint (highest validation balanced accuracy) is saved to
    ``training.checkpoint_dir/best.pt``.
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
    print(f"[train] seed: {seed}")

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
    loaders = get_dataloaders(config_path, csv_path=csv_path, artifact_root=artifact_root)

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
    # Scheduler — exponential decay applied per batch                     #
    # ------------------------------------------------------------------ #
    decay_rate  = s_cfg["decay_rate"]
    decay_steps = s_cfg["decay_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: decay_rate ** (step / decay_steps),
    )

    # ------------------------------------------------------------------ #
    # AMP scaler — scales loss to prevent float16 underflow               #
    # ------------------------------------------------------------------ #
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    # ------------------------------------------------------------------ #
    # Checkpoint                                                           #
    # ------------------------------------------------------------------ #
    checkpoint_dir = Path(t_cfg.get("checkpoint_dir", "models/layer_a"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    best_bal_acc = 0.0

    # ------------------------------------------------------------------ #
    # Early stopping                                                       #
    # ------------------------------------------------------------------ #
    patience         = t_cfg.get("early_stopping_patience", 3)
    epochs_no_improve = 0

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #
    global_step = 0

    for epoch in range(1, t_cfg["epochs"] + 1):

        train_metrics, global_step = train_one_epoch(
            model, loaders["train"], optimizer, criterion,
            device, scheduler, scaler, epoch, global_step, use_wandb=use_wandb,
        )

        val_metrics = evaluate(model, loaders["validation"], device)

        if use_wandb:
            wandb.log(
                {
                    "epoch":                      epoch,
                    "train/epoch_loss":           train_metrics["loss"],
                    "train/epoch_accuracy":       train_metrics["accuracy"],
                    "val/loss":                   val_metrics["loss"],
                    "val/accuracy":               val_metrics["accuracy"],
                    "val/balanced_accuracy":      val_metrics["balanced_accuracy"],
                    "val/binary_accuracy":        val_metrics["binary_accuracy"],
                    "val/binary_precision":       val_metrics["binary_precision"],
                    "val/binary_recall":          val_metrics["binary_recall"],
                    "val/binary_f1":              val_metrics["binary_f1"],
                    "val/binary_roc_auc":         val_metrics["binary_roc_auc"],
                },
                step=global_step,
            )

        print(
            f"Epoch {epoch:>3}/{t_cfg['epochs']}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}  "
            f"val_bin_f1={val_metrics['binary_f1']:.4f}  "
            f"val_roc_auc={val_metrics['binary_roc_auc']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc      = val_metrics["balanced_accuracy"]
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":                  epoch,
                    "model_state_dict":       model.state_dict(),
                    "optimizer_state_dict":   optimizer.state_dict(),
                    "scheduler_state_dict":   scheduler.state_dict(),
                    "scaler_state_dict":      scaler.state_dict(),
                    "best_balanced_accuracy": best_bal_acc,
                    "config":                 cfg,
                    "rng_state":              torch.get_rng_state(),
                    "cuda_rng_state":         torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                best_checkpoint_path,
            )
            if use_wandb:
                wandb.log({"val/best_balanced_accuracy": best_bal_acc}, step=global_step)
            print(f"  → checkpoint saved  (bal_acc={best_bal_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  → no improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"Early stopping: val balanced accuracy has not improved "
                      f"for {patience} consecutive epochs. Best: {best_bal_acc:.4f}")
                break

    if use_wandb:
        wandb.finish()
    return model

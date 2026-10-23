"""
Training Loop & Optimization Engine for Baseline Collision-Risk MLP.
Phase 2: Baseline Predictive Engine.

Features:
- Inverse class frequency weighting in CrossEntropyLoss to address severe class imbalance.
- AdamW optimizer with weight decay regularization and CosineAnnealingLR / ReduceLROnPlateau.
- Early Stopping monitoring validation loss to avoid overfitting.
- Automated hardware acceleration: utilizes Apple Silicon MPS or CUDA if available.
- Checkpointing saving best weights to `checkpoints/baseline_mlp.pth`.
- Training history serialization and learning curves plotting.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

# Ensure src and root are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_and_preprocess_data, ProcessedDataBundle, DEFAULT_FEATURE_COLUMNS
from src.model_baseline import CollisionRiskMLP


def get_optimal_device() -> torch.device:
    """Select the best available compute device: MPS (Apple Silicon), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Runs a single training epoch and returns (mean_loss, accuracy, macro_f1)."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        targets = batch_y.detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    n_samples = len(all_targets)
    epoch_loss = total_loss / n_samples
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, Dict[int, float]]:
    """Runs evaluation on validation or test split and computes loss, acc, and F1."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        targets = batch_y.detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    n_samples = len(all_targets)
    epoch_loss = total_loss / n_samples
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    # Per-class F1 scores
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_dict = {c: float(score) for c, score in enumerate(per_class_f1)}

    return epoch_loss, epoch_acc, epoch_f1, per_class_dict


def plot_and_save_training_curves(
    history: Dict[str, list],
    save_path: str = "results/training_curves.png",
) -> None:
    """Generates dual-panel plot of training vs validation loss and Macro F1."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # Panel 1: Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss (Weighted)", color="#3b82f6", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (Weighted)", color="#ef4444", linewidth=2, linestyle="--")
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_loss = np.min(history["val_loss"])
    ax1.scatter([best_epoch], [best_loss], color="#dc2626", s=70, zorder=5, label=f"Best Val ({best_loss:.3f} @ Ep {best_epoch})")
    ax1.set_title("Cross-Entropy Loss Trajectory", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Weighted Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Accuracy & Macro F1
    ax2.plot(epochs, history["val_macro_f1"], label="Val Macro F1", color="#10b981", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#8b5cf6", linewidth=1.5, linestyle=":")
    if "val_high_risk_f1" in history:
        ax2.plot(epochs, history["val_high_risk_f1"], label="Val Class 2 (High Risk) F1", color="#f59e0b", linewidth=2)
    ax2.set_title("Validation Classification Metrics", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score [0 - 1]", fontsize=11)
    ax2.set_ylim([0.0, 1.02])
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f" [VISUALIZATION] Training curves saved to {save_path}")


def train_model(
    csv_path: str = "dataset/collision_risk_dataset.csv",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    batch_size: int = 256,
    max_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
) -> Tuple[CollisionRiskMLP, ProcessedDataBundle, Dict[str, Any]]:
    """
    Orchestrates full training pipeline with weighted loss, early stopping,
    and checkpointing.
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_optimal_device()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, "baseline_mlp.pth")
    scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")

    print("=" * 74)
    print("   PHASE 2: BASELINE PREDICTIVE ENGINE (PYTORCH MLP)")
    print("=" * 74)
    print(f" Compute Device:         {device.type.upper()} ({torch.backends.mps.is_available() and 'Apple Silicon Metal'})")
    print(f" Dataset Source:         {csv_path}")
    print(f" Best Checkpoint Target: {best_model_path}")
    print(f" Max Epochs:             {max_epochs} | Early Stopping Patience: {patience}")
    print(f" Mini-Batch Size:        {batch_size} | Base Learning Rate: {lr}")
    print("-" * 74)

    # 1. Ingest and Process Data with Chronological Split
    bundle = load_and_preprocess_data(
        csv_path=csv_path,
        batch_size=batch_size,
        scaler_save_path=scaler_path,
    )

    print(f" Dataset Split:          Train={len(bundle.y_train):,} | Val={len(bundle.y_val):,} | Test={len(bundle.y_test):,}")
    print(f" Training Class Counts:  Safe(0)={bundle.class_counts_train.get(0, 0):,}, "
          f"Warn(1)={bundle.class_counts_train.get(1, 0):,}, "
          f"HighRisk(2)={bundle.class_counts_train.get(2, 0):,}")
    print(f" Computed Class Weights: Safe={bundle.class_weights[0]:.2f}, "
          f"Warn={bundle.class_weights[1]:.2f}, "
          f"HighRisk={bundle.class_weights[2]:.2f}")
    print(f" Input Features ({len(bundle.feature_names)}):     {', '.join(bundle.feature_names)}")

    # 2. Instantiate Baseline MLP Model
    model = CollisionRiskMLP(
        input_dim=len(bundle.feature_names),
        hidden_dims=[128, 64, 32],
        num_classes=3,
        dropout_p=0.25,
        activation="gelu",
    ).to(device)

    print(f" Model Architecture:     {model.hidden_dims} layers ({model.count_parameters():,} trainable params)")
    print("=" * 74)

    # 3. Setup Weighted Loss & Optimizer
    class_weights = bundle.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5)

    # 4. Training Loop State
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_high_risk_f1": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Train F1':^8} | {'Val Loss':^10} | {'Val Acc':^8} | {'Val Macro F1':^12} | {'High Risk F1':^12} | {'LR':^9}")
    print("-" * 88)

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train Epoch
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            dataloader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Validation Epoch
        val_loss, val_acc, val_f1, per_class = evaluate_epoch(
            model=model,
            dataloader=bundle.val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        high_risk_f1 = per_class.get(2, 0.0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["val_high_risk_f1"].append(high_risk_f1)

        # Early Stopping & Checkpoint Logic
        is_best = val_loss < (best_val_loss - 1e-4)
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save checkpoint state dict
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                    "val_high_risk_f1": high_risk_f1,
                    "input_dim": len(bundle.feature_names),
                    "hidden_dims": model.hidden_dims,
                    "feature_names": bundle.feature_names,
                    "class_weights": bundle.class_weights.cpu().numpy().tolist(),
                },
                best_model_path,
            )
            marker = "*"
        else:
            patience_counter += 1
            marker = " "

        print(
            f"{epoch:^5}{marker} | {train_loss:^10.4f} | {train_f1:^8.4f} | "
            f"{val_loss:^10.4f} | {val_acc:^8.4f} | {val_f1:^12.4f} | "
            f"{high_risk_f1:^12.4f} | {current_lr:^9.2e}"
        )

        if patience_counter >= patience:
            print("-" * 88)
            print(f" [EARLY STOPPING] Validation loss did not improve for {patience} consecutive epochs.")
            print(f" Restoring best model weights from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f}).")
            break

    elapsed_time = time.time() - start_time
    print("=" * 88)
    print(f" Training Concluded in {elapsed_time:.1f}s | Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")
    print(f" Checkpoint Saved: {best_model_path}")

    # Load best checkpoint into model
    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Save training history JSON and plots
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_and_save_training_curves(history, save_path=os.path.join(results_dir, "training_curves.png"))

    return model, bundle, history


if __name__ == "__main__":
    train_model(max_epochs=40, batch_size=256, lr=1e-3, patience=10)

"""
Training Pipeline for Attentive Collision-Risk GNN (Phase 3).
Handles:
- Batched graph training via PyG DataLoader.
- Inverse class frequency weighting in CrossEntropyLoss to preserve 100% High Risk recall.
- AdamW optimizer with gradient norm clipping and ReduceLROnPlateau scheduler.
- Early Stopping monitoring validation loss.
- Checkpointing best model weights to `checkpoints/gnn_interaction_model.pth`.
- Training curves plotting and history serialization.
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

# Ensure src and root are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_builder import load_and_preprocess_graphs, GraphDatasetBundle
from src.model_gnn import CollisionRiskGNN


def get_optimal_device() -> torch.device:
    """Select the best available compute device: MPS (Apple Silicon), CUDA, or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch_gnn(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Runs a single training epoch over batched graphs."""
    model.train()
    total_loss = 0.0
    total_edges = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(logits, batch.y)
        loss.backward()

        # Gradient clipping for stable GNN convergence
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        num_edges = batch.edge_index.size(1)
        total_loss += loss.item() * num_edges
        total_edges += num_edges

        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        targets = batch.y.detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    epoch_loss = total_loss / max(1, total_edges)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch_gnn(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, Dict[int, float]]:
    """Runs validation/testing evaluation over batched graphs."""
    model.eval()
    total_loss = 0.0
    total_edges = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        batch = batch.to(device)

        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(logits, batch.y)

        num_edges = batch.edge_index.size(1)
        total_loss += loss.item() * num_edges
        total_edges += num_edges

        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        targets = batch.y.detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    epoch_loss = total_loss / max(1, total_edges)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_dict = {c: float(score) for c, score in enumerate(per_class_f1)}

    return epoch_loss, epoch_acc, epoch_f1, per_class_dict


def plot_and_save_gnn_curves(
    history: Dict[str, list],
    save_path: str = "results/gnn_training_curves.png",
) -> None:
    """Generates dual-panel plot of GNN training vs validation loss and Macro F1."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # Panel 1: Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss (Weighted GNN)", color="#2563eb", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (Weighted GNN)", color="#dc2626", linewidth=2, linestyle="--")
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_loss = np.min(history["val_loss"])
    ax1.scatter([best_epoch], [best_loss], color="#991b1b", s=70, zorder=5, label=f"Best Val ({best_loss:.3f} @ Ep {best_epoch})")
    ax1.set_title("GNN Attentive Loss Trajectory", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Weighted Cross-Entropy Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Accuracy & Macro F1
    ax2.plot(epochs, history["val_macro_f1"], label="Val Macro F1", color="#059669", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#7c3aed", linewidth=1.5, linestyle=":")
    if "val_high_risk_f1" in history:
        ax2.plot(epochs, history["val_high_risk_f1"], label="Val Class 2 (High Risk) F1", color="#d97706", linewidth=2)
    ax2.set_title("GNN Validation Classification Metrics", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score [0 - 1]", fontsize=11)
    ax2.set_ylim([0.0, 1.02])
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f" [VISUALIZATION] GNN training curves saved to {save_path}")


def train_gnn(
    csv_path: str = "dataset/collision_risk_dataset.csv",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    batch_size: int = 64,
    max_epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
) -> Tuple[CollisionRiskGNN, GraphDatasetBundle, Dict[str, Any]]:
    """
    Executes full training pipeline for Attentive Collision-Risk GNN.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_optimal_device()
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_model_path = os.path.join(checkpoint_dir, "gnn_interaction_model.pth")
    scalers_path = os.path.join(checkpoint_dir, "gnn_scalers.pkl")

    print("=" * 76)
    print("   PHASE 3: ATTENTIVE GNN INTERACTION MODELING (PYTORCH GEOMETRIC)")
    print("=" * 76)
    print(f" Compute Device:         {device.type.upper()} ({torch.backends.mps.is_available() and 'Apple Silicon Metal'})")
    print(f" Dataset Source:         {csv_path}")
    print(f" Best Checkpoint Target: {best_model_path}")
    print(f" Max Epochs:             {max_epochs} | Early Stopping Patience: {patience}")
    print(f" Graph Batch Size:       {batch_size} | Base Learning Rate: {lr}")
    print("-" * 76)

    # 1. Ingest Graphs with Chronological Partitioning
    bundle = load_and_preprocess_graphs(
        csv_path=csv_path,
        batch_size=batch_size,
        scaler_save_path=scalers_path,
    )

    print(f" Graph Split:            Train={len(bundle.train_graphs):,} | Val={len(bundle.val_graphs):,} | Test={len(bundle.test_graphs):,} graphs")
    print(f" Node Feature Dim ({bundle.node_dim}):   [x, y, vx, vy, theta, speed]")
    print(f" Edge Feature Dim ({bundle.edge_dim}):   [dx, dy, dist, dvx, dvy, rel_spd, closing_spd, dca]")
    print(f" Class Weights:          Safe={bundle.class_weights[0]:.2f}, Warn={bundle.class_weights[1]:.2f}, HighRisk={bundle.class_weights[2]:.2f}")

    # 2. Instantiate CollisionRiskGNN
    model = CollisionRiskGNN(
        node_dim=bundle.node_dim,
        edge_dim=bundle.edge_dim,
        hidden_dim=64,
        edge_hidden_dim=32,
        num_classes=3,
        num_layers=2,
        heads=4,
        dropout_p=0.25,
    ).to(device)

    print(f" Model Architecture:     2 GATv2Conv Layers (4 Heads) + Topological Edge Head")
    print(f" Trainable Parameters:   {model.count_parameters():,}")
    print("=" * 76)

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
    print("-" * 90)

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Train Epoch
        train_loss, train_acc, train_f1 = train_one_epoch_gnn(
            model=model,
            dataloader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Validation Epoch
        val_loss, val_acc, val_f1, per_class = evaluate_epoch_gnn(
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

        # Early Stopping & Checkpointing
        is_best = val_loss < (best_val_loss - 1e-4)
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                    "val_high_risk_f1": high_risk_f1,
                    "node_dim": bundle.node_dim,
                    "edge_dim": bundle.edge_dim,
                    "hidden_dim": model.hidden_dim,
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
            print("-" * 90)
            print(f" [EARLY STOPPING] Validation loss did not improve for {patience} consecutive epochs.")
            print(f" Restoring best model weights from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f}).")
            break

    elapsed_time = time.time() - start_time
    print("=" * 90)
    print(f" GNN Training Concluded in {elapsed_time:.1f}s | Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")
    print(f" Checkpoint Saved: {best_model_path}")

    # Load best weights
    best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Save training history JSON and plots
    history_path = os.path.join(results_dir, "gnn_training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_and_save_gnn_curves(history, save_path=os.path.join(results_dir, "gnn_training_curves.png"))

    return model, bundle, history


if __name__ == "__main__":
    train_gnn(max_epochs=40, batch_size=64, lr=1e-3, patience=10)

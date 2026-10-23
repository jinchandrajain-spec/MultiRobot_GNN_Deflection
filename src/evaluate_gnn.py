"""
Evaluation & Benchmarking Suite for Attentive Collision-Risk GNN (Phase 3).
Performs:
- Evaluation on held-out test graphs (720 scene graphs, 7,200 canonical test pairs).
- Direct quantitative benchmarking against Phase 2 MLP Baseline.
- Detailed Classification Report (Precision, Recall, F1, Support).
- False Positive reduction analysis demonstrating the advantage of topological scene context.
- High-resolution Seaborn Confusion Matrix heatmap saved to `results/gnn_confusion_matrix.png`.
- Structured metrics export to `results/gnn_test_metrics.json`.
"""

import os
import sys
import json
import pickle
from typing import Dict, Any, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

# Ensure src and root are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_builder import load_and_preprocess_graphs, GraphDatasetBundle
from src.model_gnn import CollisionRiskGNN
from src.train_gnn import get_optimal_device

CLASS_NAMES = ["Safe (0)", "Warning (1)", "High Risk (2)"]


def plot_and_save_gnn_confusion_matrix(
    cm: np.ndarray,
    save_path: str = "results/gnn_confusion_matrix.png",
) -> None:
    """
    Renders and saves academic-quality Seaborn Confusion Matrix heatmap for GNN.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=180)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]:,})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Greens",
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=1.2,
        linecolor="#e2e8f0",
        square=True,
        ax=ax,
        cbar_kws={"label": "Normalized Recall Proportion"},
    )

    ax.set_title("Attentive GNN Collision-Risk Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("True Ground-Truth Risk Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Risk Label (CollisionRiskGNN)", fontsize=11, fontweight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f" [VISUALIZATION] GNN confusion matrix saved to {save_path}")


def evaluate_gnn(
    checkpoint_path: str = "checkpoints/gnn_interaction_model.pth",
    mlp_metrics_path: str = "results/test_metrics.json",
    csv_path: str = "dataset/collision_risk_dataset.csv",
    results_dir: str = "results",
) -> Dict[str, Any]:
    """
    Evaluates trained GNN on held-out test graphs and benchmarks directly
    against Phase 2 Baseline MLP.
    """
    device = get_optimal_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"GNN Checkpoint not found at: {checkpoint_path}. Train GNN first!")

    # 1. Load checkpoint and metadata
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    node_dim = checkpoint.get("node_dim", 6)
    edge_dim = checkpoint.get("edge_dim", 8)
    hidden_dim = checkpoint.get("hidden_dim", 64)
    best_epoch = checkpoint.get("epoch", -1)

    # 2. Ingest Test Graphs
    bundle = load_and_preprocess_graphs(csv_path=csv_path)

    # 3. Instantiate and load model weights
    model = CollisionRiskGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_classes=3,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 76)
    print("   PHASE 3: GNN MODEL EVALUATION & BENCHMARKING (TEST SET)")
    print("=" * 76)
    print(f" Model Checkpoint:       {checkpoint_path} (Trained for {best_epoch} epochs)")
    print(f" Test Graphs:            {len(bundle.test_graphs):,} graphs")
    print(f" Hardware Device:        {device.type.upper()}")
    print("-" * 76)

    # 4. Run Test Inference
    # Isolate canonical test edges (matching the exact 7,200 pairs from Phase 2 MLP)
    all_preds_canonical = []
    all_targets_canonical = []
    all_probs_canonical = []

    # Also track full bidirectional edges
    all_preds_full = []
    all_targets_full = []

    with torch.no_grad():
        for batch in bundle.test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            targets = batch.y.cpu().numpy()
            mask = batch.canonical_mask.cpu().numpy()

            all_preds_full.extend(preds)
            all_targets_full.extend(targets)

            all_preds_canonical.extend(preds[mask])
            all_targets_canonical.extend(targets[mask])
            all_probs_canonical.extend(probs[mask])

    y_true = np.array(all_targets_canonical)
    y_pred = np.array(all_preds_canonical)

    # 5. Compute Comprehensive Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = [int(np.sum(y_true == c)) for c in range(3)]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]).round(4)

    # Safety metrics
    high_risk_support = support[2]
    high_risk_detected = int(cm[2, 2])
    high_risk_missed_as_safe = int(cm[2, 0])
    high_risk_missed_as_warn = int(cm[2, 1])
    high_risk_recall = recalls[2]
    high_risk_fnr = (high_risk_missed_as_safe + high_risk_missed_as_warn) / max(1, high_risk_support)

    # False Positive Metrics (Safe true items predicted as Warning or High Risk)
    safe_support = support[0]
    safe_fp_count = int(cm[0, 1] + cm[0, 2])
    safe_fp_rate = safe_fp_count / max(1, safe_support)

    # 6. Display Academic Classification Report Table
    print(f"\n{'Class Name':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 64)
    for c in range(3):
        sup_str = f"{support[c]:,}"
        print(f"{CLASS_NAMES[c]:<16} | {precisions[c]:<10.4f} | {recalls[c]:<10.4f} | {f1s[c]:<10.4f} | {sup_str:<8}")
    print("-" * 64)
    tot_str = f"{len(y_true):,}"
    print(f"{'Macro Average':<16} | {np.mean(precisions):<10.4f} | {np.mean(recalls):<10.4f} | {macro_f1:<10.4f} | {tot_str:<8}")
    print(f"{'Weighted Average':<16} | {precision_score(y_true, y_pred, average='weighted', zero_division=0):<10.4f} | "
          f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):<10.4f} | {weighted_f1:<10.4f} | {tot_str:<8}")
    print(f"{'Overall Accuracy':<16} | {acc:<10.4f} (Balanced Acc: {bal_acc:.4f})")
    print("=" * 64)

    # 7. Comparative Benchmarking against Baseline MLP
    print("\n" + "=" * 76)
    print("   HEAD-TO-HEAD BENCHMARK: PHASE 2 MLP vs. PHASE 3 ATTENTIVE GNN")
    print("=" * 76)

    mlp_metrics = None
    if os.path.exists(mlp_metrics_path):
        with open(mlp_metrics_path, "r") as f:
            mlp_metrics = json.load(f)

    if mlp_metrics:
        mlp_acc = mlp_metrics["overall"]["accuracy"]
        mlp_macro_f1 = mlp_metrics["overall"]["macro_f1"]
        mlp_weighted_f1 = mlp_metrics["overall"]["weighted_f1"]
        mlp_hr_recall = mlp_metrics["safety_audit_class_2"]["high_risk_recall"]
        mlp_cm = np.array(mlp_metrics["confusion_matrix"])
        mlp_fp_count = int(mlp_cm[0, 1] + mlp_cm[0, 2])
        mlp_fp_rate = mlp_fp_count / mlp_metrics["per_class"]["Safe (0)"]["support"]

        print(f"{'Metric':<28} | {'Phase 2 MLP':<14} | {'Phase 3 GNN':<14} | {'Delta / Improvement':<18}")
        print("-" * 76)
        print(f"{'Overall Accuracy':<28} | {mlp_acc*100:<13.2f}% | {acc*100:<13.2f}% | {(acc - mlp_acc)*100:+6.2f}%")
        print(f"{'Macro F1-Score':<28} | {mlp_macro_f1:<14.4f} | {macro_f1:<14.4f} | {(macro_f1 - mlp_macro_f1):+7.4f}")
        print(f"{'Weighted F1-Score':<28} | {mlp_weighted_f1:<14.4f} | {weighted_f1:<14.4f} | {(weighted_f1 - mlp_weighted_f1):+7.4f}")
        print(f"{'Class 2 (High Risk) Recall':<28} | {mlp_hr_recall*100:<13.2f}% | {high_risk_recall*100:<13.2f}% | {(high_risk_recall - mlp_hr_recall)*100:+6.2f}%")
        print(f"{'False Positives (Safe -> Warn/High)':<28} | {mlp_fp_count:<14} | {safe_fp_count:<14} | {safe_fp_count - mlp_fp_count:+d} ({(safe_fp_count - mlp_fp_count)/max(1, mlp_fp_count)*100:+.1f}%)")
        print(f"{'Safe False Alarm Rate':<28} | {mlp_fp_rate*100:<13.2f}% | {safe_fp_rate*100:<13.2f}% | {(safe_fp_rate - mlp_fp_rate)*100:+6.2f}%")
        print("=" * 76)

    # 8. Render and save GNN Confusion Matrix plot
    cm_path = os.path.join(results_dir, "gnn_confusion_matrix.png")
    plot_and_save_gnn_confusion_matrix(cm, save_path=cm_path)

    # 9. Save structured JSON metrics
    metrics_summary = {
        "overall": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "total_test_samples": int(len(y_true)),
        },
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": float(precisions[c]),
                "recall": float(recalls[c]),
                "f1_score": float(f1s[c]),
                "support": int(support[c]),
            }
            for c in range(3)
        },
        "safety_audit_class_2": {
            "high_risk_support": int(high_risk_support),
            "high_risk_detected": int(high_risk_detected),
            "high_risk_recall": float(high_risk_recall),
            "high_risk_missed_as_safe": int(high_risk_missed_as_safe),
            "high_risk_missed_as_warning": int(high_risk_missed_as_warn),
            "high_risk_fnr": float(high_risk_fnr),
        },
        "false_positive_audit": {
            "safe_support": int(safe_support),
            "safe_false_positives": int(safe_fp_count),
            "safe_false_positive_rate": float(safe_fp_rate),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }

    json_path = os.path.join(results_dir, "gnn_test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f" [DATA EXPORT] GNN test metrics saved to {json_path}")

    return metrics_summary


if __name__ == "__main__":
    evaluate_gnn()

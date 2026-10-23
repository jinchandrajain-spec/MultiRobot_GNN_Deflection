"""
Evaluation & Quantitative Metrics Suite for Baseline Collision-Risk MLP.
Phase 2: Baseline Predictive Engine.

Features:
- Loads checkpoint `checkpoints/baseline_mlp.pth` and evaluated strictly on Test split.
- Computes Overall Accuracy, Macro F1, Weighted F1, and Balanced Accuracy.
- Generates detailed per-class Classification Report (Precision, Recall, F1-Score, Support).
- Computes critical safety metrics: High Risk (Class 2) Recall & False Negative Rate.
- Generates and saves high-resolution Confusion Matrix heatmap to `results/confusion_matrix.png`.
- Exports JSON report of all metrics to `results/test_metrics.json`.
"""

import os
import sys
import json
import pickle
from typing import Dict, Any, Tuple

import torch
import numpy as np
import pandas as pd
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

# Ensure src and root are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_and_preprocess_data, DEFAULT_FEATURE_COLUMNS
from src.model_baseline import CollisionRiskMLP
from src.train_model import get_optimal_device


CLASS_NAMES = ["Safe (0)", "Warning (1)", "High Risk (2)"]


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    save_path: str = "results/confusion_matrix.png",
    normalize: bool = True,
) -> None:
    """
    Renders and saves academic-quality Seaborn Confusion Matrix heatmap.
    Shows both percentage (row normalized) and raw sample counts.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=180)

    # Annotate with both percentage and raw counts: "94.2%\n(3,840)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j] * 100:.1f}%\n({cm[i, j]:,})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=1.2,
        linecolor="#e2e8f0",
        square=True,
        ax=ax,
        cbar_kws={"label": "Normalized Recall Proportion"},
    )

    ax.set_title("Multi-Robot Collision-Risk Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("True Ground-Truth Risk Label", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Risk Label (Baseline MLP)", fontsize=11, fontweight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f" [VISUALIZATION] Confusion matrix saved to {save_path}")


def evaluate_model(
    checkpoint_path: str = "checkpoints/baseline_mlp.pth",
    scaler_path: str = "checkpoints/scaler.pkl",
    csv_path: str = "dataset/collision_risk_dataset.csv",
    results_dir: str = "results",
) -> Dict[str, Any]:
    """
    Loads saved model checkpoint and scaler, performs test inference,
    computes rigorous classification metrics, and exports reports and charts.
    """
    device = get_optimal_device()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first!")

    # 1. Load checkpoint and metadata
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_names = checkpoint.get("feature_names", DEFAULT_FEATURE_COLUMNS)
    hidden_dims = checkpoint.get("hidden_dims", [128, 64, 32])
    best_epoch = checkpoint.get("epoch", -1)

    # 2. Ingest Test Data using standard pipeline
    bundle = load_and_preprocess_data(csv_path=csv_path, feature_cols=feature_names)

    # 3. Instantiate and load model weights
    model = CollisionRiskMLP(
        input_dim=len(feature_names),
        hidden_dims=hidden_dims,
        num_classes=3,
        dropout_p=0.25,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("=" * 74)
    print("   PHASE 2: MODEL EVALUATION & METRICS (TEST SET)")
    print("=" * 74)
    print(f" Model Checkpoint:       {checkpoint_path} (Trained for {best_epoch} epochs)")
    print(f" Test Set Size:          {len(bundle.y_test):,} samples")
    print(f" Hardware Device:        {device.type.upper()}")
    print("-" * 74)

    # 4. Run Test Inference
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in bundle.test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            all_probs.extend(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # 5. Compute Comprehensive Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    support = [int(np.sum(y_true == c)) for c in range(3)]

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]).round(4)

    # Critical Safety Metric: Class 2 (High Risk) Recall and Miss Rate (False Negatives)
    high_risk_support = support[2]
    high_risk_detected = int(cm[2, 2])
    high_risk_missed_as_safe = int(cm[2, 0])
    high_risk_missed_as_warn = int(cm[2, 1])
    high_risk_recall = recalls[2]
    high_risk_fnr = (high_risk_missed_as_safe + high_risk_missed_as_warn) / max(1, high_risk_support)

    # 6. Display Academic Classification Report Table
    print(f"\n{'Class Name':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 62)
    for c in range(3):
        sup_str = f"{support[c]:,}"
        print(f"{CLASS_NAMES[c]:<16} | {precisions[c]:<10.4f} | {recalls[c]:<10.4f} | {f1s[c]:<10.4f} | {sup_str:<8}")
    print("-" * 62)
    tot_str = f"{len(y_true):,}"
    print(f"{'Macro Average':<16} | {np.mean(precisions):<10.4f} | {np.mean(recalls):<10.4f} | {macro_f1:<10.4f} | {tot_str:<8}")
    print(f"{'Weighted Average':<16} | {precision_score(y_true, y_pred, average='weighted', zero_division=0):<10.4f} | "
          f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):<10.4f} | {weighted_f1:<10.4f} | {tot_str:<8}")
    print(f"{'Overall Accuracy':<16} | {acc:<10.4f} (Balanced Acc: {bal_acc:.4f})")
    print("=" * 62)

    print("\n--- SAFETY CRITICAL AUDIT: CLASS 2 (HIGH RISK) ---")
    print(f" Total True High Risk Encounters:  {high_risk_support:,}")
    print(f" Correctly Detected as High Risk: {high_risk_detected:,} ({high_risk_recall * 100:.2f}%)")
    print(f" Classified as Warning (Acceptable): {high_risk_missed_as_warn:,}")
    print(f" Missed as Safe (Critical Hazard):   {high_risk_missed_as_safe:,} ({high_risk_missed_as_safe / max(1, high_risk_support) * 100:.2f}%)")
    print(f" High Risk Recall:                {high_risk_recall:.4f}")
    print(f" High Risk False Negative Rate:   {high_risk_fnr:.4f}")
    print("-" * 62)

    # 7. Render and save Confusion Matrix plot
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_and_save_confusion_matrix(cm, save_path=cm_path)

    # 8. Save structured JSON metrics
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
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
    }

    json_path = os.path.join(results_dir, "test_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f" [DATA EXPORT] Test metrics saved to {json_path}")

    return metrics_summary


if __name__ == "__main__":
    evaluate_model()

"""
Data Processing Pipeline for Multi-Robot Collision-Risk Prediction (Phase 2).
Handles:
- Dataset ingestion via Pandas and NumPy.
- Isolation of relative kinematic features and distance metrics (Input X).
- Chronological temporal splitting into Train (70%), Validation (15%), Test (15%)
  to prevent temporal autocorrelation and data leakage.
- Scikit-learn StandardScaler fitted strictly on the Training set.
- PyTorch Dataset and DataLoader construction.
- Inverse class frequency calculation for weighted CrossEntropyLoss.
"""

import os
import pickle
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Default Feature Schema: Pairwise relative kinematics and distance metrics
DEFAULT_FEATURE_COLUMNS = [
    "rel_pos_x",               # Delta X position (x_j - x_i)
    "rel_pos_y",               # Delta Y position (y_j - y_i)
    "euclidean_distance",      # Center-to-center distance ||dp||
    "rel_vel_x",               # Delta Vx velocity (vx_j - vx_i)
    "rel_vel_y",               # Delta Vy velocity (vy_j - vy_i)
    "rel_speed",               # Magnitude of relative velocity ||dv||
    "closing_speed",           # Projection of relative velocity along position vector
    "closest_approach_dist",   # Projected minimum distance of closest approach (DCA)
    "speed_i",                 # Scalar speed of robot i
    "speed_j",                 # Scalar speed of robot j
]

TARGET_COLUMN = "risk_label"  # 0: Safe, 1: Warning, 2: High Risk


class CollisionRiskDataset(Dataset):
    """
    PyTorch Dataset wrapper for multi-robot tabular collision risk features.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


@dataclass
class ProcessedDataBundle:
    """Container holding DataLoader instances, raw arrays, scaler, and metadata."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    class_weights: torch.Tensor
    feature_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_counts_train: Dict[int, int]


def load_and_preprocess_data(
    csv_path: str,
    feature_cols: Optional[List[str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 128,
    scaler_save_path: Optional[str] = None,
) -> ProcessedDataBundle:
    """
    Loads raw CSV dataset, performs temporal split per scenario to prevent data leakage,
    standardizes features via training statistics, and constructs PyTorch DataLoaders.

    Args:
        csv_path: Path to dataset CSV file.
        feature_cols: List of column names to isolate as features X.
        train_ratio: Fraction for training set (default 0.70).
        val_ratio: Fraction for validation set (default 0.15).
        test_ratio: Fraction for testing set (default 0.15).
        batch_size: Mini-batch size for DataLoaders.
        scaler_save_path: Optional path to persist the fitted StandardScaler as a pickle file.

    Returns:
        ProcessedDataBundle containing DataLoaders, arrays, weights, and scaler.
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLUMNS

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Validate required columns
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Features missing from CSV: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from CSV.")

    # Chronological Split by Scenario:
    # Within each scenario sequence, split first 70% to Train, next 15% to Val, last 15% to Test.
    # This prevents temporal leakage between consecutive frames while preserving all scenarios.
    train_dfs = []
    val_dfs = []
    test_dfs = []

    scenario_groups = df.groupby("scenario", sort=False)
    for scen_name, group in scenario_groups:
        n_group = len(group)
        n_train = int(n_group * train_ratio)
        n_val = int(n_group * val_ratio)

        train_dfs.append(group.iloc[:n_train])
        val_dfs.append(group.iloc[n_train:n_train + n_val])
        test_dfs.append(group.iloc[n_train + n_val:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # Extract Raw Features and Labels
    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[TARGET_COLUMN].values.astype(np.int64)

    X_val_raw = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df[TARGET_COLUMN].values.astype(np.int64)

    X_test_raw = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[TARGET_COLUMN].values.astype(np.int64)

    # Fit StandardScaler strictly on the training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Optionally persist the scaler for downstream evaluation and inference
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        with open(scaler_save_path, "wb") as f:
            pickle.dump(scaler, f)

    # Compute Inverse Class Frequency Weights for Imbalanced Loss:
    # w_c = N_total / (N_classes * N_c)
    classes, counts = np.unique(y_train, return_counts=True)
    class_counts_train = dict(zip(classes.tolist(), counts.tolist()))
    total_train_samples = len(y_train)
    num_classes = len(classes)

    weights = np.zeros(num_classes, dtype=np.float32)
    for c, count in class_counts_train.items():
        weights[c] = total_train_samples / (num_classes * count)

    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Build PyTorch Datasets and DataLoaders
    train_dataset = CollisionRiskDataset(X_train, y_train)
    val_dataset = CollisionRiskDataset(X_val, y_val)
    test_dataset = CollisionRiskDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return ProcessedDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        class_weights=class_weights_tensor,
        feature_names=feature_cols,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_counts_train=class_counts_train,
    )

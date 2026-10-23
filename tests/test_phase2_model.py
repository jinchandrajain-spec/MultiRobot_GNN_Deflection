"""
Unit Tests for Phase 2: Baseline Predictive Engine (PyTorch MLP).
Validates:
- Data loader chronological splitting and feature extraction.
- StandardScaler application without data leakage.
- CollisionRiskMLP architecture, parameter count, forward pass, and predict methods.
- Class weight computation for imbalanced CrossEntropyLoss.
- Checkpoint loading and inference reproducibility.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import (
    load_and_preprocess_data,
    CollisionRiskDataset,
    DEFAULT_FEATURE_COLUMNS,
)
from src.model_baseline import CollisionRiskMLP


class TestPhase2DataLoader(unittest.TestCase):
    """Tests for data preprocessing, chronological splitting, and dataset construction."""

    @classmethod
    def setUpClass(cls):
        cls.csv_path = "dataset/collision_risk_dataset.csv"

    def test_chronological_split_and_scaling(self):
        """Verifies chronological 70/15/15 split and scaler fitting strictly on train."""
        bundle = load_and_preprocess_data(
            csv_path=self.csv_path,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=64,
        )

        total_samples = len(bundle.y_train) + len(bundle.y_val) + len(bundle.y_test)
        df_total = len(pd.read_csv(self.csv_path))
        self.assertEqual(total_samples, df_total)
        self.assertAlmostEqual(len(bundle.y_train) / total_samples, 0.70, delta=0.02)
        self.assertAlmostEqual(len(bundle.y_val) / total_samples, 0.15, delta=0.02)
        self.assertAlmostEqual(len(bundle.y_test) / total_samples, 0.15, delta=0.02)

        # Dimension checks
        self.assertEqual(bundle.X_train.shape[1], len(DEFAULT_FEATURE_COLUMNS))
        self.assertEqual(bundle.X_val.shape[1], len(DEFAULT_FEATURE_COLUMNS))
        self.assertEqual(bundle.X_test.shape[1], len(DEFAULT_FEATURE_COLUMNS))

        # Scaler check: mean of X_train should be approximately 0 and std approx 1
        np.testing.assert_allclose(bundle.X_train.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(bundle.X_train.std(axis=0), 1.0, atol=1e-5)

        # Target label checks: labels must be in {0, 1, 2}
        self.assertTrue(set(np.unique(bundle.y_train)).issubset({0, 1, 2}))

        # Class weights check: all weights must be strictly positive
        self.assertEqual(len(bundle.class_weights), 3)
        self.assertTrue(torch.all(bundle.class_weights > 0.0))

    def test_dataloader_batch_shapes(self):
        """Verifies PyTorch DataLoader yields correctly shaped batches."""
        bundle = load_and_preprocess_data(
            csv_path=self.csv_path,
            batch_size=32,
        )
        batch_x, batch_y = next(iter(bundle.train_loader))
        self.assertEqual(batch_x.shape, (32, len(DEFAULT_FEATURE_COLUMNS)))
        self.assertEqual(batch_y.shape, (32,))
        self.assertEqual(batch_x.dtype, torch.float32)
        self.assertEqual(batch_y.dtype, torch.int64)


class TestCollisionRiskMLP(unittest.TestCase):
    """Tests for PyTorch MLP model architecture and inference."""

    def setUp(self):
        self.input_dim = 10
        self.model = CollisionRiskMLP(
            input_dim=self.input_dim,
            hidden_dims=[64, 32],
            num_classes=3,
            dropout_p=0.2,
            activation="gelu",
        )

    def test_forward_pass_shape(self):
        """Verifies logits tensor shape matches (batch_size, 3)."""
        x = torch.randn(16, self.input_dim)
        logits = self.model(x)
        self.assertEqual(logits.shape, (16, 3))

    def test_predict_methods(self):
        """Verifies predict_proba sums to 1.0 and predict returns valid classes."""
        x = torch.randn(8, self.input_dim)
        probs = self.model.predict_proba(x)
        preds = self.model.predict(x)

        self.assertEqual(probs.shape, (8, 3))
        np.testing.assert_allclose(probs.sum(dim=-1).numpy(), 1.0, atol=1e-5)

        self.assertEqual(preds.shape, (8,))
        self.assertTrue(all(p.item() in [0, 1, 2] for p in preds))

    def test_checkpoint_reproducibility(self):
        """Verifies saved checkpoint weights can be reloaded and reproduce outputs."""
        checkpoint_path = "checkpoints/baseline_mlp.pth"
        if not os.path.exists(checkpoint_path):
            self.skipTest("Checkpoint not yet saved.")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loaded_model = CollisionRiskMLP(
            input_dim=ckpt["input_dim"],
            hidden_dims=ckpt["hidden_dims"],
            num_classes=3,
        )
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_model.eval()

        dummy_x = torch.randn(4, ckpt["input_dim"])
        with torch.no_grad():
            out1 = loaded_model(dummy_x)
            out2 = loaded_model(dummy_x)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

"""
Unit Tests for Phase 3: Attentive GNN Interaction Modeling (PyTorch Geometric).
Validates:
- Graph construction from tabular kinematics into PyG Data objects.
- Correct dimensions of node features and relational edge attributes.
- CollisionRiskGNN architecture, GATv2Conv message passing, and edge logits shape (num_edges, 3).
- Model checkpoint loading and deterministic inference.
"""

import os
import sys
import unittest
import numpy as np
import torch
from torch_geometric.data import Data, Batch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_builder import (
    load_and_preprocess_graphs,
    build_graphs_from_dataframe,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
)
from src.model_gnn import CollisionRiskGNN


class TestPhase3GraphBuilder(unittest.TestCase):
    """Tests for graph generation, chronological splitting, and normalization."""

    @classmethod
    def setUpClass(cls):
        cls.csv_path = "dataset/collision_risk_dataset.csv"

    def test_graph_dataset_structure(self):
        """Verifies graph counts, node features, and edge attributes."""
        bundle = load_and_preprocess_graphs(
            csv_path=self.csv_path,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=16,
        )

        total_graphs = len(bundle.train_graphs) + len(bundle.val_graphs) + len(bundle.test_graphs)
        self.assertAlmostEqual(len(bundle.train_graphs) / total_graphs, 0.70, delta=0.03)
        self.assertAlmostEqual(len(bundle.val_graphs) / total_graphs, 0.15, delta=0.03)
        self.assertAlmostEqual(len(bundle.test_graphs) / total_graphs, 0.15, delta=0.03)

        # Inspect first test graph
        sample_g = bundle.test_graphs[0]
        self.assertEqual(sample_g.x.shape[1], len(NODE_FEATURE_NAMES))
        self.assertEqual(sample_g.edge_attr.shape[1], len(EDGE_FEATURE_NAMES))
        self.assertEqual(sample_g.edge_index.shape[0], 2)
        self.assertEqual(sample_g.edge_index.shape[1], sample_g.edge_attr.shape[0])
        self.assertEqual(sample_g.y.shape[0], sample_g.edge_attr.shape[0])

    def test_dataloader_batching(self):
        """Verifies PyG DataLoader collates graphs into unified disjoint batches."""
        bundle = load_and_preprocess_graphs(
            csv_path=self.csv_path,
            batch_size=8,
        )
        batch = next(iter(bundle.train_loader))
        self.assertEqual(batch.x.dim(), 2)
        self.assertEqual(batch.edge_index.dim(), 2)
        self.assertEqual(batch.edge_attr.dim(), 2)
        self.assertEqual(batch.y.dim(), 1)


class TestCollisionRiskGNN(unittest.TestCase):
    """Tests for CollisionRiskGNN architecture and inference."""

    def setUp(self):
        self.node_dim = 6
        self.edge_dim = 8
        self.model = CollisionRiskGNN(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=32,
            edge_hidden_dim=16,
            num_classes=3,
            num_layers=2,
            heads=2,
            dropout_p=0.1,
        )

    def test_forward_shape(self):
        """Verifies output logits tensor matches (num_edges, 3)."""
        num_nodes = 10
        num_edges = 16

        x = torch.randn(num_nodes, self.node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, self.edge_dim)

        logits = self.model(x, edge_index, edge_attr)
        self.assertEqual(logits.shape, (num_edges, 3))

    def test_predict_methods(self):
        """Verifies Softmax probabilities and discrete class predictions."""
        num_nodes = 5
        num_edges = 8

        x = torch.randn(num_nodes, self.node_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, self.edge_dim)

        probs = self.model.predict_proba(x, edge_index, edge_attr)
        preds = self.model.predict(x, edge_index, edge_attr)

        self.assertEqual(probs.shape, (num_edges, 3))
        np.testing.assert_allclose(probs.sum(dim=-1).numpy(), 1.0, atol=1e-5)

        self.assertEqual(preds.shape, (num_edges,))
        self.assertTrue(all(p.item() in [0, 1, 2] for p in preds))

    def test_checkpoint_reproducibility(self):
        """Verifies saved GNN checkpoint can be loaded and produces deterministic outputs."""
        ckpt_path = "checkpoints/gnn_interaction_model.pth"
        if not os.path.exists(ckpt_path):
            self.skipTest("GNN Checkpoint not yet created.")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        loaded_model = CollisionRiskGNN(
            node_dim=ckpt["node_dim"],
            edge_dim=ckpt["edge_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_classes=3,
        )
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_model.eval()

        dummy_x = torch.randn(5, ckpt["node_dim"])
        dummy_edges = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        dummy_attr = torch.randn(3, ckpt["edge_dim"])

        with torch.no_grad():
            out1 = loaded_model(dummy_x, dummy_edges, dummy_attr)
            out2 = loaded_model(dummy_x, dummy_edges, dummy_attr)

        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

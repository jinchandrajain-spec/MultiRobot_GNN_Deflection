"""
Targeted Unit Tests for Phase 4 Stage B: GNN Online Inference Controller.
Validates:
- Single load of weights and scalers.
- Pairwise edge prediction structure, probability simplex, and risk category mapping.
- Per-robot threat aggregation and nearest threat attribution.
- Configurable inference cadence (frequency throttling and cached frame serving).
- Fleet scalability (4-robot and 5-robot multi-agent scenes).
- Integration with simulated scenarios and robot risk state synchronization.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    RISK_LABEL_SAFE,
    RISK_LABEL_WARNING,
    RISK_LABEL_HIGH_RISK,
    RISK_NAMES,
)
from robot_state import Robot, ScenarioManager
from src.gnn_inference_controller import (
    GNNInferenceController,
    EdgeRiskPrediction,
    RobotRiskSummary,
    InferenceFrameResult,
)


class TestGNNInferenceController(unittest.TestCase):
    """Test suite for GNN online inference controller."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_path = "checkpoints/gnn_interaction_model.pth"
        cls.scalers_path = "checkpoints/gnn_scalers.pkl"
        cls.controller = GNNInferenceController(
            checkpoint_path=cls.checkpoint_path,
            scalers_path=cls.scalers_path,
            inference_interval=1,
            device=torch.device("cpu"),  # CPU for deterministic fast unit test
        )

    def test_single_load_and_initialization(self):
        """Verifies checkpoint and scalers are loaded once and model is in eval mode."""
        self.assertFalse(self.controller.model.training)
        self.assertEqual(self.controller.checkpoint_epoch, 40)
        self.assertEqual(self.controller.node_dim, 6)
        self.assertEqual(self.controller.edge_dim, 8)
        self.assertIsNotNone(self.controller.node_scaler)
        self.assertIsNotNone(self.controller.edge_scaler)

    def test_5_robot_pairwise_prediction_structure(self):
        """Verifies output format for standard 5-robot fleet (20 directed edges)."""
        robots, _ = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_4WAY_CROSS, num_robots=5)
        result = self.controller.predict_frame(robots, frame_id=0, sim_time=0.0, force_inference=True)

        self.assertIsInstance(result, InferenceFrameResult)
        self.assertTrue(result.is_fresh_inference)
        self.assertGreater(result.inference_latency_ms, 0.0)

        # 5 robots have 5 * 4 = 20 directed edges
        self.assertEqual(len(result.edge_predictions), 20)
        self.assertEqual(len(result.robot_summaries), 5)

        for edge in result.edge_predictions:
            self.assertIsInstance(edge, EdgeRiskPrediction)
            self.assertIn(edge.risk_label, [RISK_LABEL_SAFE, RISK_LABEL_WARNING, RISK_LABEL_HIGH_RISK])
            self.assertEqual(edge.risk_category, RISK_NAMES[edge.risk_label])
            self.assertEqual(len(edge.probabilities), 3)

            # Probabilities sum to 1.0
            prob_sum = sum(edge.probabilities)
            self.assertAlmostEqual(prob_sum, 1.0, places=4)
            self.assertAlmostEqual(edge.confidence, max(edge.probabilities), places=4)
            self.assertGreaterEqual(edge.distance, 0.0)

        # Verify all robot IDs are present in summaries
        for r in robots:
            self.assertIn(r.id, result.robot_summaries)
            summary = result.robot_summaries[r.id]
            self.assertIsInstance(summary, RobotRiskSummary)
            self.assertIn(summary.max_risk_label, [0, 1, 2])
            self.assertEqual(summary.max_risk_category, RISK_NAMES[summary.max_risk_label])

    def test_4_robot_scalability(self):
        """Verifies graph generation and inference scales correctly to 4 robots (12 edges)."""
        robots, _ = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_HEAD_ON, num_robots=4)
        result = self.controller.predict_frame(robots, frame_id=1, force_inference=True)

        self.assertEqual(len(result.edge_predictions), 12)
        self.assertEqual(len(result.robot_summaries), 4)

    def test_configurable_inference_frequency_and_caching(self):
        """
        Verifies inference interval throttling:
        - When interval=2: frame 0 -> fresh, frame 1 -> cached, frame 2 -> fresh.
        """
        controller = GNNInferenceController(
            checkpoint_path=self.checkpoint_path,
            scalers_path=self.scalers_path,
            inference_interval=2,
            device=torch.device("cpu"),
        )
        robots, _ = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_4WAY_CROSS, num_robots=5)

        # Frame 0: Even frame -> Fresh inference
        res0 = controller.predict_frame(robots, frame_id=0, sim_time=0.0)
        self.assertTrue(res0.is_fresh_inference)

        # Frame 1: Odd frame -> Cached inference
        res1 = controller.predict_frame(robots, frame_id=1, sim_time=0.016)
        self.assertFalse(res1.is_fresh_inference)
        self.assertEqual(res1.frame_id, 1)

        # Frame 2: Even frame -> Fresh inference
        res2 = controller.predict_frame(robots, frame_id=2, sim_time=0.033)
        self.assertTrue(res2.is_fresh_inference)

        # Frame 3 with force_inference -> Fresh inference
        res3 = controller.predict_frame(robots, frame_id=3, sim_time=0.050, force_inference=True)
        self.assertTrue(res3.is_fresh_inference)

        metrics = controller.get_metrics()
        self.assertEqual(metrics["total_calls"], 4)
        self.assertEqual(metrics["total_model_inferences"], 3)
        self.assertEqual(metrics["total_cache_hits"], 1)
        self.assertAlmostEqual(metrics["cache_hit_rate_pct"], 25.0)

    def test_robot_state_synchronization(self):
        """Verifies update_robot_risk_states populates telemetry fields on Robot instances."""
        robots, _ = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_4WAY_CROSS, num_robots=5)
        result = self.controller.predict_frame(robots, frame_id=10, force_inference=True)

        self.controller.update_robot_risk_states(robots, result)

        for r in robots:
            expected_summary = result.robot_summaries[r.id]
            self.assertEqual(r.risk_label, expected_summary.max_risk_label)
            self.assertEqual(r.nearest_threat_id, expected_summary.highest_threat_id)
            self.assertEqual(r.nearest_distance, expected_summary.highest_threat_distance)

    def test_degenerate_single_robot(self):
        """Verifies graceful handling when fewer than 2 robots exist."""
        single_robot = [Robot(robot_id=0, x=100.0, y=100.0, target_x=200.0, target_y=200.0)]
        result = self.controller.predict_frame(single_robot, frame_id=0)

        self.assertEqual(len(result.edge_predictions), 0)
        self.assertIn(0, result.robot_summaries)
        self.assertEqual(result.robot_summaries[0].max_risk_label, RISK_LABEL_SAFE)


if __name__ == "__main__":
    unittest.main()

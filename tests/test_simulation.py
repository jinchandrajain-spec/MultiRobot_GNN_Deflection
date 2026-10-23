"""
Unit and Integration Tests for Phase 1: Intelligent Collision-Risk Prediction.
Validates:
- Analytical Time-To-Collision (TTC) calculations under various geometric conditions.
- Risk classification labeling (0: Safe, 1: Warning, 2: High Risk).
- Robot kinematics, steering, and boundary reflection physics.
- Data collection engine, CSV schema integrity, and Pandas loading.
"""

import os
import math
import shutil
import unittest
import numpy as np
import pandas as pd

from config import (
    ARENA_X,
    ARENA_Y,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    ROBOT_RADIUS,
    RISK_LABEL_SAFE,
    RISK_LABEL_WARNING,
    RISK_LABEL_HIGH_RISK,
)
from robot_state import (
    Robot,
    StaticObstacle,
    ScenarioManager,
    resolve_robot_pairwise_collisions,
)
from data_collector import (
    compute_analytical_ttc,
    classify_risk_tier,
    DataCollector,
)


class TestAnalyticalTTC(unittest.TestCase):
    """Rigorous mathematical tests for the closed-form analytical TTC solver."""

    def test_head_on_collision_exact_ttc(self):
        """
        Robot 1 at x=100, vx=+50, r=20
        Robot 2 at x=400, vx=-50, r=20
        Distance = 300. Combined radius R = 40.
        Distance to contact = 300 - 40 = 260.
        Relative closing speed = 50 - (-50) = 100.
        Expected TTC = 260 / 100 = 2.60 seconds.
        """
        ttc, dca, closing_spd = compute_analytical_ttc(
            x1=100.0, y1=200.0, vx1=50.0, vy1=0.0, r1=20.0,
            x2=400.0, y2=200.0, vx2=-50.0, vy2=0.0, r2=20.0,
        )
        self.assertAlmostEqual(ttc, 2.60, places=4)
        self.assertAlmostEqual(closing_spd, 100.0, places=4)
        self.assertAlmostEqual(dca, 0.0, places=4)

    def test_diverging_robots_infinite_ttc(self):
        """Two robots moving away from each other along X-axis."""
        ttc, _, closing_spd = compute_analytical_ttc(
            x1=100.0, y1=200.0, vx1=-50.0, vy1=0.0, r1=20.0,
            x2=400.0, y2=200.0, vx2=+50.0, vy2=0.0, r2=20.0,
        )
        self.assertTrue(math.isinf(ttc))
        self.assertLess(closing_spd, 0.0)

    def test_parallel_motion_infinite_ttc(self):
        """Two robots moving in parallel with identical velocities."""
        ttc, _, _ = compute_analytical_ttc(
            x1=100.0, y1=100.0, vx1=60.0, vy1=0.0, r1=20.0,
            x2=100.0, y2=300.0, vx2=60.0, vy2=0.0, r2=20.0,
        )
        self.assertTrue(math.isinf(ttc))

    def test_perpendicular_miss(self):
        """
        Perpendicular paths that miss each other:
        Robot 1 at (0, 0) moving East at 100.
        Robot 2 at (1000, 200) moving South at 100.
        Distance of closest approach > 20 + 20.
        """
        ttc, dca, _ = compute_analytical_ttc(
            x1=0.0, y1=0.0, vx1=100.0, vy1=0.0, r1=20.0,
            x2=1000.0, y2=500.0, vx2=0.0, vy2=100.0, r2=20.0,
        )
        self.assertTrue(math.isinf(ttc))
        self.assertGreater(dca, 40.0)

    def test_overlapping_agents_immediate_zero_ttc(self):
        """Two robots already touching or overlapping."""
        ttc, _, _ = compute_analytical_ttc(
            x1=100.0, y1=100.0, vx1=10.0, vy1=0.0, r1=20.0,
            x2=130.0, y2=100.0, vx2=-10.0, vy2=0.0, r2=20.0,
        )
        # Distance = 30, R = 40 => overlapping
        self.assertEqual(ttc, 0.0)


class TestRiskClassification(unittest.TestCase):
    """Verify risk classification tiers according to project specifications."""

    def test_safe_classification(self):
        label, cat = classify_risk_tier(4.5)
        self.assertEqual(label, RISK_LABEL_SAFE)
        self.assertEqual(cat, "SAFE")

        label_inf, _ = classify_risk_tier(float("inf"))
        self.assertEqual(label_inf, RISK_LABEL_SAFE)

    def test_warning_classification(self):
        label, cat = classify_risk_tier(2.5)
        self.assertEqual(label, RISK_LABEL_WARNING)
        self.assertEqual(cat, "WARNING")

        label_edge, _ = classify_risk_tier(3.0)
        self.assertEqual(label_edge, RISK_LABEL_WARNING)

    def test_high_risk_classification(self):
        label, cat = classify_risk_tier(0.8)
        self.assertEqual(label, RISK_LABEL_HIGH_RISK)
        self.assertEqual(cat, "HIGH_RISK")

        label_zero, _ = classify_risk_tier(0.0)
        self.assertEqual(label_zero, RISK_LABEL_HIGH_RISK)


class TestRobotKinematicsAndBoundaries(unittest.TestCase):
    """Verify kinematics, goal-seeking steering, and boundary reflection."""

    def test_goal_seeking_movement(self):
        """Robot should accelerate toward target destination."""
        r = Robot(robot_id=0, x=100.0, y=100.0, target_x=300.0, target_y=100.0)
        dt = 0.1
        initial_x = r.x
        for _ in range(10):
            r.update_kinematics(dt)
        self.assertGreater(r.x, initial_x)
        self.assertGreater(r.vx, 0.0)

    def test_boundary_collision_clamping(self):
        """Robot placed outside left boundary should be clamped inside with inverted velocity."""
        r = Robot(robot_id=1, x=ARENA_X - 10.0, y=ARENA_Y + 100.0, target_x=500.0, target_y=500.0)
        r.vx = -80.0
        collided = r.handle_boundary_collision()
        self.assertTrue(collided)
        self.assertGreaterEqual(r.x, ARENA_X + r.radius)
        self.assertGreater(r.vx, 0.0)  # Inverted

    def test_static_obstacle_collision(self):
        """Robot contacting static obstacle should be repelled."""
        obs = StaticObstacle(id=1, x=300.0, y=300.0, radius=40.0)
        r = Robot(robot_id=2, x=300.0, y=330.0, target_x=300.0, target_y=200.0, radius=20.0)
        r.vy = -50.0
        collided = r.handle_obstacle_collision(obs)
        self.assertTrue(collided)
        dist = math.hypot(r.x - obs.x, r.y - obs.y)
        self.assertGreaterEqual(dist, obs.radius + r.radius - 1e-4)

    def test_inter_robot_collision_separation(self):
        """Two overlapping robots should be separated elastically."""
        r1 = Robot(robot_id=0, x=200.0, y=200.0, target_x=300.0, target_y=200.0, radius=20.0)
        r2 = Robot(robot_id=1, x=220.0, y=200.0, target_x=100.0, target_y=200.0, radius=20.0)
        # Distance = 20, combined radius = 40 => 20px overlap
        resolve_robot_pairwise_collisions([r1, r2])
        new_dist = math.hypot(r2.x - r1.x, r2.y - r1.y)
        self.assertGreaterEqual(new_dist, 39.5)


class TestDataCollectorAndCSVOutput(unittest.TestCase):
    """Verify data collection pipeline, CSV output formatting, and dataframe loading."""

    def setUp(self):
        self.test_dir = "test_dataset_temp"
        self.test_file = "test_run.csv"
        self.collector = DataCollector(dataset_dir=self.test_dir, filename=self.test_file, auto_record=True)

    def tearDown(self):
        self.collector.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_data_logging_and_pandas_export(self):
        """Run 10 frames of 4-Way Cross and verify CSV generation and schema."""
        robots, obstacles = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_4WAY_CROSS, num_robots=5)
        dt = 1.0 / 60.0

        for frame in range(1, 15):
            sim_time = frame * dt
            for r in robots:
                r.update_kinematics(dt)
            self.collector.process_and_log_frame(frame, sim_time, "4-Way Cross", robots)

        self.collector.flush()
        self.assertTrue(os.path.exists(self.collector.filepath))

        df = self.collector.load_dataframe()
        self.assertGreater(len(df), 0)

        # Expected number of pairwise combinations for 5 robots = 5 * 4 / 2 = 10 rows per frame
        expected_cols = self.collector.CSV_HEADERS
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Verify risk labels are in {0, 1, 2}
        unique_labels = set(df["risk_label"].unique())
        self.assertTrue(unique_labels.issubset({0, 1, 2}))

        # Verify absolute coordinates and velocities exist
        self.assertTrue(np.all(np.isfinite(df["x_i"])))
        self.assertTrue(np.all(np.isfinite(df["y_i"])))
        self.assertTrue(np.all(np.isfinite(df["vx_i"])))
        self.assertTrue(np.all(np.isfinite(df["vy_i"])))


if __name__ == "__main__":
    unittest.main()

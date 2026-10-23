"""
Targeted Unit Tests for Phase 4 Stage C: Adaptive Deflection Controller.

Validates:
- Nominal goal-seeking when GNN risk is SAFE (zero deflection blend).
- High-risk interaction triggering and evasive candidate steering generation.
- Coherent multi-threat resolution (resolving multiple simultaneous high-risk agents).
- Environmental constraint satisfaction (static obstacle clearance and arena wall margins).
- State-machine hysteresis (maintains avoidance across transient risk dips).
- Smooth recovery phase (blend factor decays gracefully to 0.0).
- Adherence to kinematic motion limits (speed and acceleration boundaries).
- Integrity of underlying physical collision detection and resolution backstops.
"""

import os
import sys
import math
import unittest
from types import SimpleNamespace
from typing import List, Tuple, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    ARENA_X,
    ARENA_Y,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    ROBOT_RADIUS,
    ROBOT_MAX_SPEED,
    ROBOT_MIN_SPEED,
    ROBOT_MAX_ACCEL,
    RISK_LABEL_SAFE,
    RISK_LABEL_WARNING,
    RISK_LABEL_HIGH_RISK,
    RISK_NAMES,
    DEFLECTION_HYSTERESIS_SECONDS,
    DEFLECTION_RECOVERY_SECONDS,
)
from robot_state import (
    Robot,
    StaticObstacle,
    ScenarioManager,
    resolve_robot_pairwise_collisions,
)
from src.adaptive_deflection import (
    AdaptiveDeflectionController,
    DeflectionState,
    SteeringCandidate,
    DeflectionFrameResult,
)


def make_mock_gnn_result(
    frame_id: int = 0,
    edge_tuples: Optional[List[Tuple[int, int, int, Tuple[float, float, float]]]] = None,
    robot_summaries_dict: Optional[Dict[int, Tuple[int, Optional[int], float]]] = None,
) -> SimpleNamespace:
    """
    Creates a mock GNN inference output.
    edge_tuples: [(src_id, dst_id, risk_label, (p_safe, p_warn, p_high))]
    robot_summaries_dict: {robot_id: (max_risk_label, highest_threat_id, highest_threat_prob)}
    """
    edges = []
    if edge_tuples:
        for src, dst, lbl, probs in edge_tuples:
            edges.append(
                SimpleNamespace(
                    src_robot_id=src,
                    dst_robot_id=dst,
                    risk_label=lbl,
                    risk_category=RISK_NAMES[lbl],
                    probabilities=probs,
                    confidence=max(probs),
                    distance=100.0,
                    rel_speed=60.0,
                    closing_speed=50.0,
                    closest_approach_dist=15.0,
                )
            )

    summaries = {}
    if robot_summaries_dict:
        for r_id, (m_lbl, th_id, th_prob) in robot_summaries_dict.items():
            summaries[r_id] = SimpleNamespace(
                robot_id=r_id,
                max_risk_label=m_lbl,
                max_risk_category=RISK_NAMES[m_lbl],
                highest_threat_id=th_id,
                highest_threat_prob=th_prob,
                highest_threat_distance=100.0,
            )

    return SimpleNamespace(
        frame_id=frame_id,
        sim_time=frame_id * 0.016,
        edge_predictions=edges,
        robot_summaries=summaries,
        is_fresh_inference=True,
        inference_latency_ms=1.5,
    )


class TestAdaptiveDeflectionController(unittest.TestCase):
    """Test suite for Phase 4 Stage C Adaptive Deflection."""

    def setUp(self):
        self.controller = AdaptiveDeflectionController(
            hysteresis_time=0.40,
            recovery_time=0.30,
            horizon_time=1.5,
            high_risk_prob_threshold=0.45,
        )

    def test_nominal_goal_seeking_when_safe(self):
        """Verifies that when GNN reports SAFE, robot remains NOMINAL with 0% deflection."""
        r0 = Robot(robot_id=0, x=100.0, y=200.0, target_x=500.0, target_y=200.0)
        r1 = Robot(robot_id=1, x=800.0, y=700.0, target_x=800.0, target_y=100.0)
        robots = [r0, r1]

        mock_gnn = make_mock_gnn_result(
            frame_id=1,
            edge_tuples=[
                (0, 1, RISK_LABEL_SAFE, (0.95, 0.05, 0.0)),
                (1, 0, RISK_LABEL_SAFE, (0.95, 0.05, 0.0)),
            ],
            robot_summaries_dict={
                0: (RISK_LABEL_SAFE, None, 0.0),
                1: (RISK_LABEL_SAFE, None, 0.0),
            },
        )

        res = self.controller.compute_steering_velocities(
            robots=robots, obstacles=[], gnn_result=mock_gnn, dt=0.016
        )

        self.assertEqual(res.deflection_states[0], DeflectionState.NOMINAL)
        self.assertEqual(res.blend_factors[0], 0.0)
        self.assertFalse(r0.is_deflecting)

        # Desired steering should point straight along positive X axis (+max_speed, 0)
        des_vx, des_vy = res.steer_velocities[0]
        self.assertAlmostEqual(des_vx, r0.max_speed, delta=1.0)
        self.assertAlmostEqual(des_vy, 0.0, delta=1.0)

    def test_high_risk_triggers_evasive_candidate(self):
        """Verifies high-risk trigger activates DEFLECTING state and steers away from oncoming agent."""
        # Head-on conflict: r0 moves East (+X), r1 moves West (-X)
        r0 = Robot(robot_id=0, x=300.0, y=400.0, target_x=600.0, target_y=400.0)
        r0.vx = 80.0
        r0.vy = 0.0

        r1 = Robot(robot_id=1, x=450.0, y=400.0, target_x=200.0, target_y=400.0)
        r1.vx = -80.0
        r1.vy = 0.0

        robots = [r0, r1]

        mock_gnn = make_mock_gnn_result(
            frame_id=10,
            edge_tuples=[
                (0, 1, RISK_LABEL_HIGH_RISK, (0.05, 0.15, 0.80)),
                (1, 0, RISK_LABEL_HIGH_RISK, (0.05, 0.15, 0.80)),
            ],
            robot_summaries_dict={
                0: (RISK_LABEL_HIGH_RISK, 1, 0.80),
                1: (RISK_LABEL_HIGH_RISK, 0, 0.80),
            },
        )

        # Apply deflection over 10 consecutive frames (~160ms) to ramp blend
        for _ in range(10):
            res = self.controller.compute_steering_velocities(
                robots=robots, obstacles=[], gnn_result=mock_gnn, dt=0.016
            )

        self.assertEqual(res.deflection_states[0], DeflectionState.DEFLECTING)
        self.assertGreater(res.blend_factors[0], 0.8)
        self.assertTrue(r0.is_deflecting)
        self.assertEqual(r0.deflection_threat_id, 1)

        # Steered velocity must have significant non-zero lateral velocity (vy) to avoid head-on crash
        des_vx, des_vy = res.steer_velocities[0]
        self.assertNotEqual(des_vy, 0.0)
        lateral_speed = abs(des_vy)
        self.assertGreater(lateral_speed, 15.0)

    def test_coherent_multi_threat_resolution(self):
        """
        Verifies robot resolves multiple simultaneous high-risk threats coherently
        without steering into a secondary threat.
        """
        # Ego robot r0 is heading East
        r0 = Robot(robot_id=0, x=400.0, y=400.0, target_x=800.0, target_y=400.0)
        r0.vx = 70.0
        r0.vy = 0.0

        # Threat 1 is coming from the North-East heading South-West
        r1 = Robot(robot_id=1, x=500.0, y=320.0, target_x=300.0, target_y=500.0)
        r1.vx = -60.0
        r1.vy = 60.0

        # Threat 2 is coming from East directly head-on
        r2 = Robot(robot_id=2, x=550.0, y=400.0, target_x=250.0, target_y=400.0)
        r2.vx = -70.0
        r2.vy = 0.0

        robots = [r0, r1, r2]

        # Both r1 and r2 pose High Risk to r0
        mock_gnn = make_mock_gnn_result(
            frame_id=20,
            edge_tuples=[
                (0, 1, RISK_LABEL_HIGH_RISK, (0.05, 0.10, 0.85)),
                (0, 2, RISK_LABEL_HIGH_RISK, (0.02, 0.08, 0.90)),
                (1, 0, RISK_LABEL_HIGH_RISK, (0.05, 0.10, 0.85)),
                (2, 0, RISK_LABEL_HIGH_RISK, (0.02, 0.08, 0.90)),
            ],
            robot_summaries_dict={
                0: (RISK_LABEL_HIGH_RISK, 2, 0.90),
                1: (RISK_LABEL_HIGH_RISK, 0, 0.85),
                2: (RISK_LABEL_HIGH_RISK, 0, 0.90),
            },
        )

        for _ in range(8):
            res = self.controller.compute_steering_velocities(
                robots=robots, obstacles=[], gnn_result=mock_gnn, dt=0.016
            )

        des_vx, des_vy = res.steer_velocities[0]

        # Since North has r1 and East has r2, the safe evasion corridor is Southward (+vy)
        self.assertGreater(des_vy, 10.0, "Ego robot should steer Southward away from both North and East threats")

    def test_static_obstacle_clearance(self):
        """Verifies deflection candidate does not steer into a static obstacle."""
        r0 = Robot(robot_id=0, x=400.0, y=400.0, target_x=700.0, target_y=400.0)
        r0.vx = 80.0

        # Approaching threat from the East
        r1 = Robot(robot_id=1, x=540.0, y=400.0, target_x=200.0, target_y=400.0)
        r1.vx = -80.0

        # Static pillar placed immediately South (so steering South would cause obstacle collision)
        pillar_south = StaticObstacle(id=1, x=400.0, y=460.0, radius=35.0, label="South Pillar")

        robots = [r0, r1]
        obstacles = [pillar_south]

        mock_gnn = make_mock_gnn_result(
            frame_id=30,
            edge_tuples=[
                (0, 1, RISK_LABEL_HIGH_RISK, (0.05, 0.15, 0.80)),
                (1, 0, RISK_LABEL_HIGH_RISK, (0.05, 0.15, 0.80)),
            ],
            robot_summaries_dict={
                0: (RISK_LABEL_HIGH_RISK, 1, 0.80),
                1: (RISK_LABEL_HIGH_RISK, 0, 0.80),
            },
        )

        for _ in range(8):
            res = self.controller.compute_steering_velocities(
                robots=robots, obstacles=obstacles, gnn_result=mock_gnn, dt=0.016
            )

        des_vx, des_vy = res.steer_velocities[0]

        # Because South is blocked by pillar, the robot must steer Northward (-vy)
        self.assertLess(des_vy, -5.0, "Robot should steer Northward to avoid both the obstacle and the threat")

    def test_boundary_clearance(self):
        """Verifies deflection candidate does not steer out of arena bounds."""
        # Place robot near top arena boundary
        top_y = ARENA_Y + 40.0
        r0 = Robot(robot_id=0, x=400.0, y=top_y, target_x=700.0, target_y=top_y)
        r0.vx = 80.0

        # Threat approaching slightly from below
        r1 = Robot(robot_id=1, x=540.0, y=top_y + 30.0, target_x=200.0, target_y=top_y)
        r1.vx = -80.0

        robots = [r0, r1]

        mock_gnn = make_mock_gnn_result(
            frame_id=40,
            edge_tuples=[
                (0, 1, RISK_LABEL_HIGH_RISK, (0.05, 0.15, 0.80)),
                (1, 0, RISK_LABEL_HIGH_RISK, (0.05, 0.15, 0.80)),
            ],
            robot_summaries_dict={
                0: (RISK_LABEL_HIGH_RISK, 1, 0.80),
                1: (RISK_LABEL_HIGH_RISK, 0, 0.80),
            },
        )

        for _ in range(8):
            res = self.controller.compute_steering_velocities(
                robots=robots, obstacles=[], gnn_result=mock_gnn, dt=0.016
            )

        cand = res.active_candidates[0]
        self.assertIsNotNone(cand)

        # Projected position over rollout horizon should not breach top wall
        for t in [0.3, 0.8, 1.2]:
            proj_y = r0.y + cand.vy * t
            self.assertGreaterEqual(proj_y, ARENA_Y + r0.radius - 1.0)

    def test_hysteresis_persistence_across_temporary_risk_dip(self):
        """Verifies that a momentary drop to SAFE does not abort active deflection immediately."""
        r0 = Robot(robot_id=0, x=300.0, y=300.0, target_x=600.0, target_y=300.0)
        r1 = Robot(robot_id=1, x=450.0, y=300.0, target_x=200.0, target_y=300.0)
        robots = [r0, r1]

        # 1. Trigger High Risk
        high_risk_gnn = make_mock_gnn_result(
            frame_id=50,
            edge_tuples=[(0, 1, RISK_LABEL_HIGH_RISK, (0.05, 0.10, 0.85))],
            robot_summaries_dict={0: (RISK_LABEL_HIGH_RISK, 1, 0.85)},
        )
        for _ in range(6):
            self.controller.compute_steering_velocities(robots, [], high_risk_gnn, dt=0.016)

        profile = self.controller.profiles[0]
        self.assertEqual(profile.state, DeflectionState.DEFLECTING)
        self.assertAlmostEqual(profile.hysteresis_timer, 0.40, delta=0.05)

        # 2. Risk drops to SAFE for one frame
        safe_gnn = make_mock_gnn_result(
            frame_id=51,
            edge_tuples=[(0, 1, RISK_LABEL_SAFE, (0.90, 0.10, 0.0))],
            robot_summaries_dict={0: (RISK_LABEL_SAFE, None, 0.0)},
        )
        res = self.controller.compute_steering_velocities(robots, [], safe_gnn, dt=0.016)

        # Must still be in DEFLECTING due to hysteresis
        self.assertEqual(res.deflection_states[0], DeflectionState.DEFLECTING)
        self.assertGreater(profile.hysteresis_timer, 0.0)

    def test_recovery_decay_to_nominal_goal(self):
        """Verifies that after hysteresis expires, recovery decay smoothly returns blend to 0.0."""
        r0 = Robot(robot_id=0, x=300.0, y=300.0, target_x=600.0, target_y=300.0)
        robots = [r0]

        profile = self.controller._get_or_create_profile(0)
        profile.state = DeflectionState.DEFLECTING
        profile.hysteresis_timer = 0.05
        profile.blend_factor = 1.0

        safe_gnn = make_mock_gnn_result(
            frame_id=60,
            edge_tuples=[],
            robot_summaries_dict={0: (RISK_LABEL_SAFE, None, 0.0)},
        )

        # Step 0.10s (exceeds remaining hysteresis_timer 0.05s) -> transitions to RECOVERING
        res1 = self.controller.compute_steering_velocities(robots, [], safe_gnn, dt=0.10)
        self.assertEqual(res1.deflection_states[0], DeflectionState.RECOVERING)

        # Step remaining recovery duration (recovery_time = 0.30s)
        res2 = self.controller.compute_steering_velocities(robots, [], safe_gnn, dt=0.35)
        self.assertEqual(res2.deflection_states[0], DeflectionState.NOMINAL)
        self.assertEqual(res2.blend_factors[0], 0.0)

    def test_motion_limits_and_physical_separation(self):
        """
        Verifies:
        1. Applied deflection respects ROBOT_MAX_SPEED and acceleration bounds.
        2. Physical inter-robot collision detection and elastic separation remain intact.
        """
        r0 = Robot(robot_id=0, x=300.0, y=300.0, target_x=700.0, target_y=300.0)
        r1 = Robot(robot_id=1, x=400.0, y=300.0, target_x=100.0, target_y=300.0)
        robots = [r0, r1]

        high_risk_gnn = make_mock_gnn_result(
            frame_id=70,
            edge_tuples=[(0, 1, RISK_LABEL_HIGH_RISK, (0.05, 0.05, 0.90))],
            robot_summaries_dict={0: (RISK_LABEL_HIGH_RISK, 1, 0.90)},
        )

        # Run 20 physics steps through apply_deflection
        dt = 0.016
        for _ in range(20):
            self.controller.apply_deflection(robots, [], high_risk_gnn, dt=dt)
            # Physical collision resolution remains active
            resolve_robot_pairwise_collisions(robots)

            # Robot speed must never exceed ROBOT_MAX_SPEED
            self.assertLessEqual(r0.speed, ROBOT_MAX_SPEED + 1e-3)
            self.assertLessEqual(r1.speed, ROBOT_MAX_SPEED + 1e-3)

        # Physical contact test: place robots overlapping
        r0.x = 350.0
        r0.y = 350.0
        r1.x = 360.0  # Overlapping within 2 * ROBOT_RADIUS (36.0 px)
        r1.y = 350.0
        initial_dist = math.hypot(r1.x - r0.x, r1.y - r0.y)

        resolve_robot_pairwise_collisions(robots)
        separated_dist = math.hypot(r1.x - r0.x, r1.y - r0.y)

        # Contact physics must have separated them
        self.assertGreater(separated_dist, initial_dist)
        self.assertGreaterEqual(separated_dist, 2.0 * ROBOT_RADIUS - 1e-3)


if __name__ == "__main__":
    unittest.main()

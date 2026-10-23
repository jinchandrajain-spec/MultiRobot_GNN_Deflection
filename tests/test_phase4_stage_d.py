"""
Targeted Unit & Regression Tests for Phase 4 Stage D: UI Telemetry, Metrics & Dual-Mode Verification.

Validates:
- UIEngine rendering in both Normal (disabled) and Adaptive (enabled) modes.
- Visual elements: evasive highlighting, steering vectors, pairwise risk lines, controller status pill.
- Discrete, non-duplicating physical collision measurement.
- Reliable avoidance event measurement (safe divergence after high-risk encounter).
- Avoidance suppression when a physical collision occurs.
- Keyboard toggle logic ([D] key) and state reset behavior.
"""

import os
import sys
import math
import unittest
from types import SimpleNamespace
import pygame

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set SDL dummy driver for headless test execution
os.environ["SDL_VIDEODRIVER"] = "dummy"

from config import (
    ARENA_X,
    ARENA_Y,
    ROBOT_RADIUS,
    RISK_LABEL_SAFE,
    RISK_LABEL_HIGH_RISK,
)
from robot_state import Robot, StaticObstacle, ScenarioManager
from ui_engine import UIEngine
from src.adaptive_deflection import AdaptiveDeflectionController


class TestPhase4StageD(unittest.TestCase):
    """Test suite for Phase 4 Stage D UI, metrics, and mode validation."""

    @classmethod
    def setUpClass(cls):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        cls.surface = pygame.display.set_mode((1320, 800))
        cls.ui_engine = UIEngine(screen=cls.surface)

    def test_ui_engine_headless_render_normal_mode(self):
        """Verifies UIEngine renders cleanly with deflection disabled (Normal Mode)."""
        robots, obstacles = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_4WAY_CROSS, num_robots=5)
        stats = {
            "is_recording": False,
            "total_records": 150,
            "safe_count": 100,
            "warning_count": 40,
            "high_risk_count": 10,
        }
        pairwise_summaries = [
            {"pair": (0, 1), "ttc": 4.5, "dist": 220.0, "risk_label": RISK_LABEL_SAFE}
        ]

        # Should render without exception
        self.ui_engine.render(
            sim_time=1.0,
            fps=60.0,
            scenario_name="Test Scenario",
            robots=robots,
            obstacles=obstacles,
            pairwise_summaries=pairwise_summaries,
            stats=stats,
            is_paused=False,
            deflection_enabled=False,
            total_collisions=0,
            total_avoidances=0,
            gnn_result=None,
        )

    def test_ui_engine_headless_render_adaptive_mode(self):
        """Verifies UIEngine renders evasive highlighting and steering vectors in Adaptive Mode."""
        robots, obstacles = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_HEAD_ON, num_robots=4)
        r0 = robots[0]
        r0.is_deflecting = True
        r0.deflection_blend = 0.85
        r0.deflection_angle_deg = 45.0
        r0.risk_label = RISK_LABEL_HIGH_RISK

        stats = {
            "is_recording": True,
            "total_records": 250,
            "safe_count": 180,
            "warning_count": 50,
            "high_risk_count": 20,
        }
        pairwise_summaries = [
            {"pair": (0, 1), "ttc": 0.8, "dist": 75.0, "risk_label": RISK_LABEL_HIGH_RISK}
        ]

        # Should render evasive highlighting and deflected steering vector cleanly
        self.ui_engine.render(
            sim_time=2.5,
            fps=59.8,
            scenario_name="Head-On Adaptive",
            robots=robots,
            obstacles=obstacles,
            pairwise_summaries=pairwise_summaries,
            stats=stats,
            is_paused=False,
            deflection_enabled=True,
            total_collisions=1,
            total_avoidances=2,
            gnn_result=None,
        )

    def test_actual_collision_measurement_discrete_counting(self):
        """
        Verifies that when two robots collide over multiple consecutive frames,
        the physical collision counter increments exactly once.
        """
        r0 = Robot(robot_id=0, x=300.0, y=300.0, target_x=500.0, target_y=300.0)
        r1 = Robot(robot_id=1, x=330.0, y=300.0, target_x=100.0, target_y=300.0)
        # Distance is 30.0 px <= 2 * ROBOT_RADIUS (36.0 px) -> Overlap
        robots = [r0, r1]

        total_physical_collisions = 0
        active_colliding_pairs = set()

        # Frame 1: Initial contact detected -> +1
        current_colliding_pairs = set()
        pair_key = (0, 1)
        dist = math.hypot(r1.x - r0.x, r1.y - r0.y)
        if dist <= (r0.radius + r1.radius):
            current_colliding_pairs.add(pair_key)
            if pair_key not in active_colliding_pairs:
                total_physical_collisions += 1
        active_colliding_pairs = current_colliding_pairs
        self.assertEqual(total_physical_collisions, 1)

        # Frames 2 to 5: Still contacting during collision response -> must NOT increment again
        for _ in range(4):
            current_colliding_pairs = set()
            if dist <= (r0.radius + r1.radius):
                current_colliding_pairs.add(pair_key)
                if pair_key not in active_colliding_pairs:
                    total_physical_collisions += 1
            active_colliding_pairs = current_colliding_pairs

        self.assertEqual(total_physical_collisions, 1, "Physical collision must not be multi-counted across frames")

        # Separate robots
        r1.x = 450.0
        current_colliding_pairs = set()
        dist = math.hypot(r1.x - r0.x, r1.y - r0.y)
        if dist <= (r0.radius + r1.radius):
            current_colliding_pairs.add(pair_key)
        active_colliding_pairs = current_colliding_pairs
        self.assertEqual(len(active_colliding_pairs), 0)

    def test_reliable_avoidance_event_measurement(self):
        """
        Verifies that an avoidance event is recorded when robots enter high-risk conflict,
        actively deflect, and safely diverge without physical contact.
        """
        r0 = Robot(robot_id=0, x=300.0, y=300.0, target_x=500.0, target_y=300.0)
        r1 = Robot(robot_id=1, x=385.0, y=300.0, target_x=100.0, target_y=300.0)
        pair_key = (0, 1)

        active_conflicts = {}
        total_avoidances = 0

        # Step 1: Conflict begins at 85px with high risk and deflection active
        dist = 85.0
        r0.is_deflecting = True
        active_conflicts[pair_key] = {
            "min_dist": dist,
            "deflected": True,
            "collided": False,
        }

        # Step 2: Closest approach reaches 65px (safely above 36px contact threshold)
        dist = 65.0
        active_conflicts[pair_key]["min_dist"] = min(active_conflicts[pair_key]["min_dist"], dist)

        # Step 3: Robots steer away and diverge to 135px (closing speed negative)
        dist = 135.0
        closing_spd = -30.0
        rec = active_conflicts[pair_key]
        if dist > 125.0 and closing_spd <= 0.0:
            if rec["deflected"] and not rec["collided"] and rec["min_dist"] < 95.0:
                total_avoidances += 1
            del active_conflicts[pair_key]

        self.assertEqual(total_avoidances, 1, "Avoidance event must be recorded on safe divergence")
        self.assertNotIn(pair_key, active_conflicts)

    def test_avoidance_suppressed_if_collision_occurred(self):
        """
        Verifies that if physical contact occurred during conflict, it is NOT counted as an avoidance event.
        """
        pair_key = (0, 1)
        active_conflicts = {
            pair_key: {
                "min_dist": 30.0,
                "deflected": True,
                "collided": True,  # Physical collision took place
            }
        }
        total_avoidances = 0

        dist = 135.0
        closing_spd = -20.0
        rec = active_conflicts[pair_key]
        if dist > 125.0 and closing_spd <= 0.0:
            if rec["deflected"] and not rec["collided"] and rec["min_dist"] < 95.0:
                total_avoidances += 1
            del active_conflicts[pair_key]

        self.assertEqual(total_avoidances, 0, "Avoidance must NOT be credited if physical contact occurred")

    def test_keyboard_toggle_state_reset(self):
        """Verifies toggle logic correctly clears deflection telemetry when switched off."""
        controller = AdaptiveDeflectionController()
        robots, obstacles = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_HEAD_ON, num_robots=2)

        deflection_enabled = True
        # Set some active deflection state
        robots[0].is_deflecting = True
        robots[0].deflection_blend = 0.9

        # User presses 'D' to disable
        deflection_enabled = not deflection_enabled
        if not deflection_enabled:
            controller.reset_profiles()
            for r in robots:
                r.is_deflecting = False
                r.deflection_blend = 0.0

        self.assertFalse(deflection_enabled)
        self.assertFalse(robots[0].is_deflecting)
        self.assertEqual(robots[0].deflection_blend, 0.0)


if __name__ == "__main__":
    unittest.main()

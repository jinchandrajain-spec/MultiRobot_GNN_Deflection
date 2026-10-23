"""
Adaptive Deflection Controller for Multi-Robot Navigation (Phase 4 Stage C).

Responsibilities:
- Connects GNN pairwise collision-risk predictions to robot kinematics and steering.
- On high-risk interactions, generates and evaluates safe candidate steering velocities.
- Coherently resolves multiple simultaneous threats without steering into secondary agents.
- Enforces static obstacle clearance and bounded arena wall safety margins.
- Preserves original goal destination trajectories whenever safe.
- Respects kinematic motion limits (ROBOT_MIN_SPEED, ROBOT_MAX_SPEED, ROBOT_MAX_ACCEL).
- Implements state-machine hysteresis (persistence) and smooth blend recovery (decay).
- Keeps physical boundary, obstacle, and inter-robot collision detection enabled.
"""

import os
import sys
import math
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

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
    DEFLECTION_HORIZON_SECONDS,
    DEFLECTION_PROJECTION_STEPS,
    DEFLECTION_ROBOT_SAFETY_MARGIN,
    DEFLECTION_OBSTACLE_SAFETY_MARGIN,
    DEFLECTION_WALL_SAFETY_MARGIN,
    DEFLECTION_HIGH_RISK_PROB_THRESH,
)
from robot_state import Robot, StaticObstacle


class DeflectionState:
    """Discrete behavioral state for an autonomous agent's deflection controller."""
    NOMINAL = "NOMINAL"        # Pure goal-seeking navigation (Safe)
    DEFLECTING = "DEFLECTING"  # Active candidate steering to evade high-risk threat
    RECOVERING = "RECOVERING"  # Threat resolved; smoothly blending back to goal


@dataclass
class SteeringCandidate:
    """Evaluated candidate velocity vector for collision avoidance."""
    angle_rad: float
    speed: float
    vx: float
    vy: float
    cost_total: float
    cost_robot_threats: float
    cost_obstacles: float
    cost_boundaries: float
    cost_goal: float
    cost_smoothness: float
    is_feasible: bool


@dataclass
class RobotDeflectionProfile:
    """Per-robot persistent state machine and hysteresis tracking."""
    robot_id: int
    state: str = DeflectionState.NOMINAL
    active_threat_ids: List[int] = field(default_factory=list)
    hysteresis_timer: float = 0.0
    recovery_timer: float = 0.0
    blend_factor: float = 0.0          # [0.0 = nominal goal, 1.0 = 100% deflection candidate]
    last_candidate: Optional[SteeringCandidate] = None
    last_steer_vx: float = 0.0
    last_steer_vy: float = 0.0
    deflection_side: int = 0           # 0 = uncommitted, +1 = right/starboard, -1 = left/port
    deflection_threat_id: Optional[int] = None
    deflected_trajectory: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class DeflectionFrameResult:
    """Aggregated deflection outputs and telemetry for a simulation time step."""
    frame_id: int
    steer_velocities: Dict[int, Tuple[float, float]]  # {robot_id: (desired_vx, desired_vy)}
    deflection_states: Dict[int, str]                # {robot_id: state}
    blend_factors: Dict[int, float]                  # {robot_id: blend}
    active_candidates: Dict[int, Optional[SteeringCandidate]]
    active_deflecting_count: int


class AdaptiveDeflectionController:
    """
    Online Adaptive Deflection Steering Engine for Multi-Robot Navigation.
    Evaluates multi-agent scene geometry, obstacles, boundaries, and GNN predictions
    to steer robots away from high-risk conflicts while preserving goal navigation.
    """

    def __init__(
        self,
        hysteresis_time: float = DEFLECTION_HYSTERESIS_SECONDS,
        recovery_time: float = DEFLECTION_RECOVERY_SECONDS,
        horizon_time: float = DEFLECTION_HORIZON_SECONDS,
        projection_steps: int = DEFLECTION_PROJECTION_STEPS,
        high_risk_prob_threshold: float = DEFLECTION_HIGH_RISK_PROB_THRESH,
        robot_safety_margin: float = DEFLECTION_ROBOT_SAFETY_MARGIN,
        obstacle_safety_margin: float = DEFLECTION_OBSTACLE_SAFETY_MARGIN,
        wall_safety_margin: float = DEFLECTION_WALL_SAFETY_MARGIN,
        w_robot: float = 8.0,
        w_obs: float = 10.0,
        w_wall: float = 10.0,
        w_goal: float = 1.2,
        w_smooth: float = 0.8,
    ):
        self.hysteresis_time = float(hysteresis_time)
        self.recovery_time = float(recovery_time)
        self.horizon_time = float(horizon_time)
        self.projection_steps = max(2, int(projection_steps))
        self.high_risk_prob_threshold = float(high_risk_prob_threshold)

        self.robot_safety_margin = float(robot_safety_margin)
        self.obstacle_safety_margin = float(obstacle_safety_margin)
        self.wall_safety_margin = float(wall_safety_margin)

        # Objective function weights
        self.w_robot = float(w_robot)
        self.w_obs = float(w_obs)
        self.w_wall = float(w_wall)
        self.w_goal = float(w_goal)
        self.w_smooth = float(w_smooth)

        # Persistent per-robot profiles
        self.profiles: Dict[int, RobotDeflectionProfile] = {}

    def _get_or_create_profile(self, robot_id: int) -> RobotDeflectionProfile:
        """Retrieves or initializes the persistent state profile for a robot."""
        if robot_id not in self.profiles:
            self.profiles[robot_id] = RobotDeflectionProfile(robot_id=robot_id)
        return self.profiles[robot_id]

    def reset_profiles(self) -> None:
        """Clear all active deflection profiles (e.g. upon scenario switch)."""
        self.profiles.clear()

    def _determine_avoidance_side(
        self,
        robot: Robot,
        threat_robot: Robot,
        obstacles: List[StaticObstacle],
        profile: RobotDeflectionProfile,
    ) -> int:
        """
        Determines and locks a consistent lateral avoidance side (+1 for right/starboard, -1 for left/port).
        Follows standard marine/robotics passing rules and obstacle clearance to eliminate chatter.
        """
        if profile.deflection_side != 0 and profile.deflection_threat_id == threat_robot.id:
            return profile.deflection_side

        dx = threat_robot.x - robot.x
        dy = threat_robot.y - robot.y
        dist = math.hypot(dx, dy)
        if dist < 1e-4:
            profile.deflection_side = 1
            profile.deflection_threat_id = threat_robot.id
            return 1

        threat_ang = math.atan2(dy, dx)
        rel_bearing = (threat_ang - robot.theta + math.pi) % (2.0 * math.pi) - math.pi

        # Probe lateral test clearances at 45-degree offset
        ang_right = robot.theta + math.radians(45.0)
        ang_left = robot.theta - math.radians(45.0)

        def eval_clearance(test_ang: float) -> float:
            tx = robot.x + math.cos(test_ang) * 65.0
            ty = robot.y + math.sin(test_ang) * 65.0
            wall_d = min(
                tx - (ARENA_X + robot.radius),
                (ARENA_X + ARENA_WIDTH - robot.radius) - tx,
                ty - (ARENA_Y + robot.radius),
                (ARENA_Y + ARENA_HEIGHT - robot.radius) - ty,
            )
            obs_d = 999.0
            for obs in obstacles:
                d = math.hypot(tx - obs.x, ty - obs.y) - (obs.radius + robot.radius)
                obs_d = min(obs_d, d)
            return min(wall_d, obs_d)

        clear_r = eval_clearance(ang_right)
        clear_l = eval_clearance(ang_left)

        if clear_r < 18.0 and clear_l >= 18.0:
            side = -1  # Right side obstructed by obstacle/wall -> deflect left
        elif clear_l < 18.0 and clear_r >= 18.0:
            side = 1   # Left side obstructed -> deflect right
        else:
            # Symmetrical or open: if near head-on (|bearing| < 40 deg), follow right-hand rule
            if abs(rel_bearing) < math.radians(40.0):
                side = 1
            elif rel_bearing > 0:
                side = -1  # Threat is on starboard -> steer port
            else:
                side = 1   # Threat is on port -> steer starboard

        profile.deflection_side = side
        profile.deflection_threat_id = threat_robot.id
        return side

    def _extract_threats_for_robot(
        self,
        robot: Any = None,
        gnn_result: Optional[Any] = None,
        robots_by_id: Optional[Dict[int, Robot]] = None,
        robot_id: Optional[int] = None,
    ) -> Tuple[bool, List[int]]:
        """
        Parses GNN prediction output and confirms credible physical threat geometry.
        Returns:
            (is_high_risk, list_of_threatening_robot_ids)
        """
        if gnn_result is None:
            return False, []

        if robot_id is None:
            if isinstance(robot, int):
                robot_id = robot
            elif hasattr(robot, "id"):
                robot_id = robot.id
            else:
                robot_id = 0

        robot_obj = robots_by_id.get(robot_id) if robots_by_id else (robot if hasattr(robot, "x") else None)

        threat_ids: List[int] = []
        is_high_risk = False

        # 1. Edge-level predictions
        edge_predictions = getattr(gnn_result, "edge_predictions", [])
        for edge in edge_predictions:
            src_id = getattr(edge, "src_robot_id", None)
            dst_id = getattr(edge, "dst_robot_id", None)
            risk_lbl = getattr(edge, "risk_label", RISK_LABEL_SAFE)
            probs = getattr(edge, "probabilities", (1.0, 0.0, 0.0))

            if src_id == robot_id:
                prob_high = probs[2] if len(probs) > 2 else 0.0
                prob_warn = probs[1] if len(probs) > 1 else 0.0
                has_model_risk = (
                    risk_lbl == RISK_LABEL_HIGH_RISK
                    or prob_high >= self.high_risk_prob_threshold
                    or (risk_lbl == RISK_LABEL_WARNING and (prob_high + prob_warn) >= 0.70)
                )

                if has_model_risk:
                    is_credible = True
                    if robot_obj is not None and robots_by_id is not None and dst_id in robots_by_id:
                        other = robots_by_id[dst_id]
                        dx = other.x - robot_obj.x
                        dy = other.y - robot_obj.y
                        dist = math.hypot(dx, dy)
                        dvx = other.vx - robot_obj.vx
                        dvy = other.vy - robot_obj.vy
                        closing_spd = -(dx * dvx + dy * dvy) / max(1e-6, dist)

                        a = dvx * dvx + dvy * dvy
                        b = 2.0 * (dx * dvx + dy * dvy)
                        t_cpa = max(0.0, -b / (2.0 * a)) if a >= 1e-8 else 0.0
                        dca = math.sqrt((dx + dvx * t_cpa) ** 2 + (dy + dvy * t_cpa) ** 2)

                        # Threat is credible if closing in within 280px and predicted approach is unsafe, or within 95px
                        is_credible = (
                            (dist < 280.0 and closing_spd > 3.0 and dca < 75.0)
                            or (dist < 95.0)
                        )

                    if is_credible:
                        is_high_risk = True
                        if dst_id is not None and dst_id not in threat_ids:
                            threat_ids.append(dst_id)

        # 2. Robot summaries fallback / confirmation (only used if no edge predictions are available, e.g. mock unit tests)
        if not edge_predictions:
            robot_summaries = getattr(gnn_result, "robot_summaries", {})
            if robot_id in robot_summaries:
                summary = robot_summaries[robot_id]
                max_lbl = getattr(summary, "max_risk_label", RISK_LABEL_SAFE)
                threat_id = getattr(summary, "highest_threat_id", None)
                threat_prob = getattr(summary, "highest_threat_prob", 0.0)

                if max_lbl == RISK_LABEL_HIGH_RISK or threat_prob >= self.high_risk_prob_threshold:
                    is_high_risk = True
                    if threat_id is not None and threat_id not in threat_ids:
                        threat_ids.append(threat_id)

        return is_high_risk, threat_ids

    def _generate_candidate_velocities(
        self,
        robot: Robot,
        goal_vx: float,
        goal_vy: float,
        threat_robots: List[Robot],
        profile: Optional[RobotDeflectionProfile] = None,
        num_angle_bins: int = 24,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Generates candidate steering vectors (angle_rad, speed, vx, vy) spanning:
        1. Direct goal progress.
        2. Current heading continuity.
        3. Smooth forward-biased lateral evasive bypass vectors adhering to the committed avoidance side.
        4. Full rotation fallback bins.
        """
        candidates: List[Tuple[float, float, float, float]] = []
        seen = set()

        def add_cand(ang: float, spd: float):
            norm_ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
            key = (round(norm_ang, 2), round(spd, 1))
            if key not in seen:
                seen.add(key)
                candidates.append((norm_ang, spd, math.cos(norm_ang) * spd, math.sin(norm_ang) * spd))

        speeds = [robot.max_speed, robot.max_speed * 0.85, robot.min_speed]

        # 1. Goal Direction
        goal_dist = math.hypot(goal_vx, goal_vy)
        ang_goal = math.atan2(goal_vy, goal_vx) if goal_dist > 1e-4 else robot.theta
        for spd in speeds:
            add_cand(ang_goal, spd)

        # 2. Current Heading Direction
        for spd in speeds[:2]:
            add_cand(robot.theta, spd)

        # 3. Forward-biased evasive steering candidates
        side = profile.deflection_side if profile and profile.deflection_side != 0 else None
        sides = [side] if side is not None else [1, -1]
        evasion_offsets_deg = [22.0, 32.0, 45.0, 60.0, 75.0, 90.0]

        for t_robot in threat_robots:
            dx = t_robot.x - robot.x
            dy = t_robot.y - robot.y
            dist = math.hypot(dx, dy)
            if dist > 1e-4:
                threat_ang = math.atan2(dy, dx)
                for s in sides:
                    for off_deg in evasion_offsets_deg:
                        off_rad = s * math.radians(off_deg)
                        cand_goal_off = ang_goal + off_rad
                        cand_threat_off = threat_ang + off_rad
                        for spd in [robot.max_speed, robot.max_speed * 0.85, robot.max_speed * 0.70]:
                            add_cand(cand_goal_off, spd)
                            add_cand(cand_threat_off, spd)

        # 4. Uniform circular discrete candidate angles for environmental recovery
        for i in range(num_angle_bins):
            ang = -math.pi + (2.0 * math.pi * i) / num_angle_bins
            for spd in [robot.max_speed, robot.min_speed]:
                add_cand(ang, spd)

        return candidates


    def _evaluate_candidate(
        self,
        candidate: Tuple[float, float, float, float],
        robot: Robot,
        goal_vx: float,
        goal_vy: float,
        all_robots: List[Robot],
        active_threat_ids: List[int],
        obstacles: List[StaticObstacle],
        profile: RobotDeflectionProfile,
    ) -> SteeringCandidate:
        """
        Evaluates safety, goal progression, and kinematic continuity for a candidate velocity.
        """
        ang, spd, cvx, cvy = candidate

        cost_threats = 0.0
        cost_obstacles = 0.0
        cost_boundaries = 0.0
        is_feasible = True

        # Generate rollout time steps along the forward horizon
        time_checkpoints = [
            0.15,
            0.30,
            0.55,
            0.85,
            1.20,
            self.horizon_time,
        ]

        # Arena boundaries with robot radius
        min_x = ARENA_X + robot.radius
        max_x = ARENA_X + ARENA_WIDTH - robot.radius
        min_y = ARENA_Y + robot.radius
        max_y = ARENA_Y + ARENA_HEIGHT - robot.radius

        for t in time_checkpoints:
            # Projected ego position
            px = robot.x + cvx * t
            py = robot.y + cvy * t

            # -------------------------------------------------------------
            # 1. Arena Boundary Clearance
            # -------------------------------------------------------------
            if px < min_x or px > max_x or py < min_y or py > max_y:
                cost_boundaries += 6000.0
                is_feasible = False
            else:
                dist_wall = min(px - min_x, max_x - px, py - min_y, max_y - py)
                if dist_wall < self.wall_safety_margin:
                    pen = (self.wall_safety_margin - dist_wall) / self.wall_safety_margin
                    cost_boundaries += (pen ** 2) * 80.0

            # -------------------------------------------------------------
            # 2. Static Obstacle Clearance
            # -------------------------------------------------------------
            for obs in obstacles:
                dx_obs = px - obs.x
                dy_obs = py - obs.y
                dist_obs_sq = dx_obs * dx_obs + dy_obs * dy_obs
                min_clear = obs.radius + robot.radius

                if dist_obs_sq <= (min_clear * min_clear):
                    # Intersects obstacle
                    cost_obstacles += 7000.0
                    is_feasible = False
                else:
                    dist_obs = math.sqrt(dist_obs_sq)
                    clearance = dist_obs - min_clear
                    if clearance < self.obstacle_safety_margin:
                        pen = (self.obstacle_safety_margin - clearance) / self.obstacle_safety_margin
                        cost_obstacles += (pen ** 2) * 100.0

            # -------------------------------------------------------------
            # 3. Multi-Robot Threat & Fleet Collision Avoidance
            # (Evaluates ALL other robots simultaneously to prevent cascading crashes)
            # -------------------------------------------------------------
            for other in all_robots:
                if other.id == robot.id:
                    continue

                # Projected other agent position (constant velocity assumption)
                other_px = other.x + other.vx * t
                other_py = other.y + other.vy * t

                dx_rob = px - other_px
                dy_rob = py - other_py
                dist_rob_sq = dx_rob * dx_rob + dy_rob * dy_rob
                min_dist_rob = robot.radius + other.radius

                # Multiplier is higher if agent is an active GNN high-risk threat
                is_active_threat = other.id in active_threat_ids
                threat_mult = 3.0 if is_active_threat else 1.0

                if dist_rob_sq <= (min_dist_rob * min_dist_rob):
                    cost_threats += 8000.0 * threat_mult
                    is_feasible = False
                else:
                    dist_rob = math.sqrt(dist_rob_sq)
                    clearance = dist_rob - min_dist_rob
                    if clearance < self.robot_safety_margin:
                        pen = (self.robot_safety_margin - clearance) / self.robot_safety_margin
                        cost_threats += (pen ** 2) * 120.0 * threat_mult

        # -----------------------------------------------------------------
        # 4. Goal Alignment (Preserve Original Mission Progress)
        # -----------------------------------------------------------------
        goal_mag = math.hypot(goal_vx, goal_vy)
        cand_mag = spd
        if goal_mag > 1e-4 and cand_mag > 1e-4:
            cos_goal = (cvx * goal_vx + cvy * goal_vy) / (cand_mag * goal_mag)
            cost_goal_align = (1.0 - cos_goal) * 0.5  # [0.0 = aligned, 1.0 = opposite]
        else:
            cost_goal_align = 0.0

        # Reward maintaining forward pace
        speed_penalty = (robot.max_speed - spd) / robot.max_speed * 0.25
        cost_goal = cost_goal_align + speed_penalty

        # -----------------------------------------------------------------
        # 5. Smoothness & Kinetic Continuity (Chatter prevention)
        # -----------------------------------------------------------------
        prev_vx = profile.last_steer_vx
        prev_vy = profile.last_steer_vy
        prev_mag = math.hypot(prev_vx, prev_vy)

        if prev_mag > 1e-4 and cand_mag > 1e-4:
            cos_smooth = (cvx * prev_vx + cvy * prev_vy) / (cand_mag * prev_mag)
            cost_smoothness = (1.0 - cos_smooth) * 1.5
        elif robot.speed > 1e-4 and cand_mag > 1e-4:
            cos_smooth = (cvx * robot.vx + cvy * robot.vy) / (cand_mag * robot.speed)
            cost_smoothness = (1.0 - cos_smooth) * 1.5
        else:
            cost_smoothness = 0.0

        # Avoidance Direction Consistency (Strict anti-chatter enforcement)
        if profile.state == DeflectionState.DEFLECTING and profile.deflection_side != 0 and active_threat_ids:
            primary_threat = next((r for r in all_robots if r.id == profile.deflection_threat_id), None)
            if primary_threat is None:
                primary_threat = next((r for r in all_robots if r.id in active_threat_ids), None)
            if primary_threat is not None:
                dx_th = primary_threat.x - robot.x
                dy_th = primary_threat.y - robot.y
                # Only enforce if oncoming threat is in front of the robot
                if (dx_th * math.cos(robot.theta) + dy_th * math.sin(robot.theta)) > 0:
                    th_ang = math.atan2(dy_th, dx_th)
                    cand_rel = (ang - th_ang + math.pi) % (2.0 * math.pi) - math.pi
                    cand_side = 1 if cand_rel > 0 else -1
                    if cand_side != profile.deflection_side and abs(cand_rel) < math.radians(130.0):
                        cost_smoothness += 1200.0

        # Total Weighted Cost
        cost_total = (
            self.w_robot * cost_threats
            + self.w_obs * cost_obstacles
            + self.w_wall * cost_boundaries
            + self.w_goal * cost_goal
            + self.w_smooth * cost_smoothness
        )

        return SteeringCandidate(
            angle_rad=ang,
            speed=spd,
            vx=cvx,
            vy=cvy,
            cost_total=cost_total,
            cost_robot_threats=cost_threats,
            cost_obstacles=cost_obstacles,
            cost_boundaries=cost_boundaries,
            cost_goal=cost_goal,
            cost_smoothness=cost_smoothness,
            is_feasible=is_feasible,
        )

    def _project_deflected_trajectory(
        self,
        robot: Robot,
        steer_vx: float,
        steer_vy: float,
        goal_vx: float,
        goal_vy: float,
        steps: int = 8,
        dt_step: float = 0.20,
    ) -> List[Tuple[float, float]]:
        """
        Projects a smooth forward evasive trajectory arc for visual path rendering.
        The curve departs along the evasive steering vector and bends gracefully back toward the destination.
        """
        waypoints: List[Tuple[float, float]] = []
        cur_x = float(robot.x)
        cur_y = float(robot.y)

        for k in range(1, steps + 1):
            alpha = min(1.0, (k / steps) ** 1.35)
            vx = (1.0 - alpha) * steer_vx + alpha * goal_vx
            vy = (1.0 - alpha) * steer_vy + alpha * goal_vy
            cur_x += vx * dt_step
            cur_y += vy * dt_step
            waypoints.append((round(cur_x, 1), round(cur_y, 1)))

        return waypoints

    def compute_steering_velocities(
        self,
        robots: List[Robot],
        obstacles: List[StaticObstacle],
        gnn_result: Optional[Any],
        dt: float,
    ) -> DeflectionFrameResult:
        """
        Computes safe steering velocities for all robots based on GNN predictions
        and multi-agent scene geometry without modifying robot positions.
        """
        frame_id = getattr(gnn_result, "frame_id", 0) if gnn_result else 0
        steer_velocities: Dict[int, Tuple[float, float]] = {}
        deflection_states: Dict[int, str] = {}
        blend_factors: Dict[int, float] = {}
        active_candidates: Dict[int, Optional[SteeringCandidate]] = {}
        deflecting_count = 0

        robots_by_id = {r.id: r for r in robots}

        for robot in robots:
            profile = self._get_or_create_profile(robot.id)

            # 1. Nominal Goal Vector
            dx = robot.target_x - robot.x
            dy = robot.target_y - robot.y
            dist_to_goal = math.hypot(dx, dy)
            if dist_to_goal > 1e-4:
                goal_vx = (dx / dist_to_goal) * robot.max_speed
                goal_vy = (dy / dist_to_goal) * robot.max_speed
            else:
                goal_vx = 0.0
                goal_vy = 0.0

            # 2. Extract GNN Threat Status with Credible Proximity Check
            is_high_risk, current_threat_ids = self._extract_threats_for_robot(
                robot=robot, gnn_result=gnn_result, robots_by_id=robots_by_id
            )

            # Check if active threats have safely passed and cleared
            if profile.state == DeflectionState.DEFLECTING:
                all_threats_passed = True
                for tid in profile.active_threat_ids:
                    if tid in robots_by_id:
                        other = robots_by_id[tid]
                        odx = other.x - robot.x
                        ody = other.y - robot.y
                        odist = math.hypot(odx, ody)
                        odvx = other.vx - robot.vx
                        odvy = other.vy - robot.vy
                        oclosing = -(odx * odvx + ody * odvy) / max(1e-6, odist)
                        if not (oclosing <= -4.0 and odist > (robot.radius + other.radius + 15.0)):
                            all_threats_passed = False
                            break
                    else:
                        all_threats_passed = False

                if all_threats_passed and profile.active_threat_ids:
                    is_high_risk = False
                    profile.hysteresis_timer = 0.0
                    profile.state = DeflectionState.RECOVERING
                    profile.recovery_timer = self.recovery_time
                    profile.active_threat_ids = []
                    profile.deflection_side = 0
                    profile.deflection_threat_id = None

            # 3. Update Deflection State Machine with Hysteresis & Recovery
            if is_high_risk:
                # Active high-risk conflict detected
                profile.state = DeflectionState.DEFLECTING
                profile.active_threat_ids = current_threat_ids
                profile.hysteresis_timer = self.hysteresis_time

                # Establish or maintain consistent avoidance side
                if current_threat_ids:
                    primary_tid = current_threat_ids[0]
                    if primary_tid in robots_by_id:
                        self._determine_avoidance_side(
                            robot=robot,
                            threat_robot=robots_by_id[primary_tid],
                            obstacles=obstacles,
                            profile=profile,
                        )

                # Rapid blend engagement (~100ms)
                profile.blend_factor = min(1.0, profile.blend_factor + (dt / 0.10))

            else:
                # No active high risk this frame
                if profile.state == DeflectionState.DEFLECTING:
                    # Decrement hysteresis hold timer
                    profile.hysteresis_timer -= dt
                    if profile.hysteresis_timer <= 0.0:
                        # Hysteresis expired -> begin smooth recovery
                        profile.state = DeflectionState.RECOVERING
                        profile.recovery_timer = self.recovery_time
                        profile.active_threat_ids = []
                        profile.deflection_side = 0
                        profile.deflection_threat_id = None

                elif profile.state == DeflectionState.RECOVERING:
                    # Decay blend factor smoothly toward nominal
                    profile.recovery_timer -= dt
                    if profile.recovery_timer <= 0.0 or self.recovery_time <= 0.0:
                        profile.state = DeflectionState.NOMINAL
                        profile.blend_factor = 0.0
                        profile.active_threat_ids = []
                        profile.deflection_side = 0
                        profile.deflection_threat_id = None
                    else:
                        profile.blend_factor = max(0.0, profile.recovery_timer / self.recovery_time)

                else:
                    # NOMINAL state
                    profile.state = DeflectionState.NOMINAL
                    profile.blend_factor = 0.0
                    profile.active_threat_ids = []
                    profile.deflection_side = 0
                    profile.deflection_threat_id = None

            # 4. Compute Steering Output
            if profile.blend_factor <= 1e-4 and profile.state == DeflectionState.NOMINAL:
                # Pure nominal navigation
                final_vx = goal_vx
                final_vy = goal_vy
                best_candidate = None
                profile.last_steer_vx = goal_vx
                profile.last_steer_vy = goal_vy
                profile.last_candidate = None

            else:
                # Evaluate Candidate Steering Rollouts
                threat_robots = [
                    robots_by_id[tid] for tid in profile.active_threat_ids if tid in robots_by_id
                ]
                raw_candidates = self._generate_candidate_velocities(
                    robot=robot,
                    goal_vx=goal_vx,
                    goal_vy=goal_vy,
                    threat_robots=threat_robots,
                    profile=profile,
                )

                best_candidate: Optional[SteeringCandidate] = None
                lowest_cost = float("inf")

                for cand in raw_candidates:
                    eval_cand = self._evaluate_candidate(
                        candidate=cand,
                        robot=robot,
                        goal_vx=goal_vx,
                        goal_vy=goal_vy,
                        all_robots=robots,
                        active_threat_ids=profile.active_threat_ids,
                        obstacles=obstacles,
                        profile=profile,
                    )
                    if eval_cand.cost_total < lowest_cost:
                        lowest_cost = eval_cand.cost_total
                        best_candidate = eval_cand

                if best_candidate is None:
                    # Fallback to goal if somehow no candidate evaluated
                    cand_vx, cand_vy = goal_vx, goal_vy
                else:
                    cand_vx, cand_vy = best_candidate.vx, best_candidate.vy

                # Smoothly blend between nominal goal velocity and safe candidate velocity
                alpha = profile.blend_factor
                final_vx = (1.0 - alpha) * goal_vx + alpha * cand_vx
                final_vy = (1.0 - alpha) * goal_vy + alpha * cand_vy

                # Clamp blended velocity to robot speed limits
                final_spd = math.hypot(final_vx, final_vy)
                if final_spd > robot.max_speed:
                    scale = robot.max_speed / final_spd
                    final_vx *= scale
                    final_vy *= scale
                elif final_spd < robot.min_speed and dist_to_goal > 1.0:
                    scale = robot.min_speed / max(1e-4, final_spd)
                    final_vx *= scale
                    final_vy *= scale

                profile.last_steer_vx = final_vx
                profile.last_steer_vy = final_vy
                profile.last_candidate = best_candidate

            # Check if meaningful lateral evasive steering adjustment is applied
            goal_spd = math.hypot(goal_vx, goal_vy)
            steer_spd = math.hypot(final_vx, final_vy)
            if goal_spd > 1e-4 and steer_spd > 1e-4:
                cos_dev = (final_vx * goal_vx + final_vy * goal_vy) / (steer_spd * goal_spd)
                cos_dev = max(-1.0, min(1.0, cos_dev))
                steer_dev_deg = math.degrees(math.acos(cos_dev))
            else:
                steer_dev_deg = 0.0

            is_genuinely_deflecting = (
                profile.state == DeflectionState.DEFLECTING
                and profile.blend_factor > 0.12
                and (steer_dev_deg > 3.0 or math.hypot(final_vx - goal_vx, final_vy - goal_vy) > 8.0)
            )

            # Update Telemetry fields on the Robot object
            robot.is_deflecting = is_genuinely_deflecting
            robot.deflection_blend = profile.blend_factor
            if math.hypot(final_vx, final_vy) > 1.0:
                robot.deflection_angle_deg = math.degrees(math.atan2(final_vy, final_vx)) % 360.0
            robot.deflection_threat_id = profile.active_threat_ids[0] if profile.active_threat_ids else None

            # Attach projected deflected trajectory for UI rendering
            if is_genuinely_deflecting:
                robot.deflected_path = self._project_deflected_trajectory(
                    robot=robot,
                    steer_vx=final_vx,
                    steer_vy=final_vy,
                    goal_vx=goal_vx,
                    goal_vy=goal_vy,
                )
            else:
                robot.deflected_path = []

            if is_genuinely_deflecting:
                deflecting_count += 1

            steer_velocities[robot.id] = (final_vx, final_vy)
            deflection_states[robot.id] = profile.state
            blend_factors[robot.id] = profile.blend_factor
            active_candidates[robot.id] = best_candidate

        return DeflectionFrameResult(
            frame_id=frame_id,
            steer_velocities=steer_velocities,
            deflection_states=deflection_states,
            blend_factors=blend_factors,
            active_candidates=active_candidates,
            active_deflecting_count=deflecting_count,
        )


    def apply_deflection(
        self,
        robots: List[Robot],
        obstacles: List[StaticObstacle],
        gnn_result: Optional[Any],
        dt: float,
    ) -> DeflectionFrameResult:
        """
        Calculates safe steering and applies it directly to each robot's kinematics update,
        preserving boundary checks, obstacle collisions, and inter-robot physics.
        """
        # 1. Compute safe steering velocities
        frame_result = self.compute_steering_velocities(
            robots=robots,
            obstacles=obstacles,
            gnn_result=gnn_result,
            dt=dt,
        )

        # 2. Update robot kinematics with the computed desired steering vectors
        for robot in robots:
            des_vx, des_vy = frame_result.steer_velocities[robot.id]
            robot.update_kinematics(dt=dt, desired_vx=des_vx, desired_vy=des_vy)

        return frame_result

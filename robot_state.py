"""
Physics & Kinematics Backend for Multi-Robot Navigation Simulation.
Phase 1: Advanced UI & Data Generation Engine.

Implements:
- Robot kinematics (position, velocity, heading, goal-seeking steering).
- Static circular and rectangular obstacles.
- Physical boundary collisions and inter-robot separation physics.
- Multi-agent benchmark scenarios for diverse risk data generation.
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from config import (
    ARENA_X,
    ARENA_Y,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    ROBOT_RADIUS,
    ROBOT_MAX_SPEED,
    ROBOT_MIN_SPEED,
    ROBOT_MAX_ACCEL,
    GOAL_REACH_DISTANCE,
    WALL_BOUNCE_RESTITUTION,
    ROBOT_COLLISION_RESTITUTION,
    ROBOT_COLORS,
    RISK_LABEL_SAFE,
    RISK_LABEL_WARNING,
    RISK_LABEL_HIGH_RISK,
)


@dataclass
class StaticObstacle:
    """
    Represents a static geometric obstacle in the bounded environment.
    Supports circular pillars with visual radius and bounding collision.
    """
    id: int
    x: float
    y: float
    radius: float
    label: str = "Pillar"

    def check_robot_collision(self, robot_x: float, robot_y: float, robot_r: float) -> Tuple[bool, float, float, float]:
        """
        Check if a robot intersects this static obstacle.
        Returns:
            (is_colliding, normal_x, normal_y, penetration_depth)
        """
        dx = robot_x - self.x
        dy = robot_y - self.y
        dist_sq = dx * dx + dy * dy
        min_dist = self.radius + robot_r

        if dist_sq < min_dist * min_dist:
            dist = math.sqrt(dist_sq)
            if dist > 1e-6:
                nx = dx / dist
                ny = dy / dist
                depth = min_dist - dist
            else:
                nx, ny, depth = 1.0, 0.0, min_dist
            return True, nx, ny, depth
        return False, 0.0, 0.0, 0.0


class Robot:
    """
    Autonomous mobile robot with 2D kinematic state, goal-directed steering,
    boundary avoidance, and real-time collision telemetry tracking.
    """

    def __init__(
        self,
        robot_id: int,
        x: float,
        y: float,
        target_x: float,
        target_y: float,
        color: Optional[Tuple[int, int, int]] = None,
        name: Optional[str] = None,
        radius: float = ROBOT_RADIUS,
        max_speed: float = ROBOT_MAX_SPEED,
        max_accel: float = ROBOT_MAX_ACCEL,
    ):
        self.id = robot_id
        self.name = name if name is not None else f"Robot-{robot_id}"
        self.color = color if color is not None else ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        self.radius = radius
        self.max_speed = max_speed
        self.min_speed = ROBOT_MIN_SPEED
        self.max_accel = max_accel

        # Spatial Coordinates & Velocity
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0

        # Orientation / Heading angle theta in radians [-pi, pi]
        # theta = 0 points East (+X), theta = pi/2 points South (+Y in screen space)
        self.theta = 0.0

        # Target Destination
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.waypoints: List[Tuple[float, float]] = []
        self.waypoint_index: int = 0

        # Risk & Telemetry State
        self.min_ttc: float = 99.9
        self.risk_label: int = RISK_LABEL_SAFE
        self.nearest_threat_id: Optional[int] = None
        self.nearest_distance: float = 999.0

        # Deflection State & Telemetry (Phase 4 Stage C)
        self.is_deflecting: bool = False
        self.deflection_blend: float = 0.0          # [0.0 = nominal, 1.0 = full deflection]
        self.deflection_angle_deg: float = 0.0      # Active steering heading in degrees
        self.deflection_threat_id: Optional[int] = None

        # Motion Trail (recent historical positions for rendering)
        self.trail: deque = deque(maxlen=28)
        self.trail.append((self.x, self.y))

        # Initial direction alignment
        self._align_heading_to_target()

    def _align_heading_to_target(self) -> None:
        """Initialize initial heading pointing toward the destination."""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        if abs(dx) > 1e-4 or abs(dy) > 1e-4:
            self.theta = math.atan2(dy, dx)
            # Give an initial gentle nudge in target direction
            initial_speed = self.min_speed
            self.vx = math.cos(self.theta) * initial_speed
            self.vy = math.sin(self.theta) * initial_speed

    @property
    def speed(self) -> float:
        """Current scalar speed in pixels per second."""
        return math.hypot(self.vx, self.vy)

    @property
    def heading_deg(self) -> float:
        """Current heading in degrees [0, 360)."""
        deg = math.degrees(self.theta) % 360.0
        return deg

    @property
    def compass_direction(self) -> str:
        """Returns standard cardinal/intercardinal direction name (N, NE, E, etc.)."""
        # Screen coordinates: 0 deg = East, 90 deg = South, 180 deg = West, 270 deg = North
        deg = self.heading_deg
        directions = ["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", 
                      "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"]
        idx = int((deg + 11.25) / 22.5) % 16
        return directions[idx]

    def set_target(self, tx: float, ty: float) -> None:
        """Update current target destination."""
        self.target_x = float(tx)
        self.target_y = float(ty)

    def set_waypoint_route(self, waypoints: List[Tuple[float, float]], loop: bool = True) -> None:
        """Assign an automated route of waypoints."""
        self.waypoints = list(waypoints)
        self.waypoint_index = 0
        self.loop_waypoints = loop
        if self.waypoints:
            self.set_target(self.waypoints[0][0], self.waypoints[0][1])

    def advance_waypoint_if_reached(self) -> bool:
        """
        Check if the robot reached its current target.
        If yes, advances to the next waypoint or picks a randomized arena point.
        """
        dist_to_target = math.hypot(self.target_x - self.x, self.target_y - self.y)
        if dist_to_target <= GOAL_REACH_DISTANCE:
            if self.waypoints and len(self.waypoints) > 1:
                self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
                next_wp = self.waypoints[self.waypoint_index]
                self.set_target(next_wp[0], next_wp[1])
                return True
            else:
                # Pick a randomized target inside the arena with safety margin
                margin = self.radius * 3.5
                rx = random.uniform(ARENA_X + margin, ARENA_X + ARENA_WIDTH - margin)
                ry = random.uniform(ARENA_Y + margin, ARENA_Y + ARENA_HEIGHT - margin)
                self.set_target(rx, ry)
                return True
        return False

    def update_kinematics(
        self,
        dt: float,
        desired_vx: Optional[float] = None,
        desired_vy: Optional[float] = None,
    ) -> None:
        """
        Update position and velocity using goal-directed proportional steering
        and bounded acceleration kinematics:
        v_desired = normalize(target - position) * max_speed (or caller provided desired steering)
        a = clamp((v_desired - v) / tau, max_accel)
        v = v + a * dt
        p = p + v * dt
        """
        # 1. Target vector
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)

        if desired_vx is not None and desired_vy is not None:
            # Caller-provided (e.g. adaptive deflection) steering guidance
            des_vx = desired_vx
            des_vy = desired_vy
            has_intent = (abs(des_vx) > 1e-4 or abs(des_vy) > 1e-4)
        elif dist > 1e-4:
            # Desired velocity vector pointing directly to target
            des_vx = (dx / dist) * self.max_speed
            des_vy = (dy / dist) * self.max_speed
            has_intent = True
        else:
            des_vx = 0.0
            des_vy = 0.0
            has_intent = False

        if has_intent:
            # Steering force / acceleration
            time_constant = 0.35  # Responsiveness parameter in seconds
            accel_x = (des_vx - self.vx) / time_constant
            accel_y = (des_vy - self.vy) / time_constant

            # Limit acceleration by max_accel
            accel_mag = math.hypot(accel_x, accel_y)
            if accel_mag > self.max_accel:
                accel_x = (accel_x / accel_mag) * self.max_accel
                accel_y = (accel_y / accel_mag) * self.max_accel

            # Integrate velocity
            self.vx += accel_x * dt
            self.vy += accel_y * dt
        else:
            # Dampen velocity if directly on target
            self.vx *= 0.90
            self.vy *= 0.90

        # Clamp speed between minimum roaming speed and max_speed
        current_spd = self.speed
        if current_spd > self.max_speed:
            scale = self.max_speed / current_spd
            self.vx *= scale
            self.vy *= scale
        elif current_spd < self.min_speed and dist > GOAL_REACH_DISTANCE:
            # Maintain minimum forward momentum
            if current_spd > 1e-4:
                scale = self.min_speed / current_spd
                self.vx *= scale
                self.vy *= scale

        # Integrate spatial position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Update heading orientation angle
        if self.speed > 1.0:
            self.theta = math.atan2(self.vy, self.vx)

        # Update historical position trail for rendering
        if len(self.trail) == 0 or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 5.0:
            self.trail.append((self.x, self.y))

    def handle_boundary_collision(self) -> bool:
        """
        Enforce boundary physics so the robot strictly stays inside the room.
        Applies elastic restitution bounce when contacting the room boundaries.
        """
        collided = False
        min_x = ARENA_X + self.radius
        max_x = ARENA_X + ARENA_WIDTH - self.radius
        min_y = ARENA_Y + self.radius
        max_y = ARENA_Y + ARENA_HEIGHT - self.radius

        # Left / Right walls
        if self.x < min_x:
            self.x = min_x
            if self.vx < 0:
                self.vx = -self.vx * WALL_BOUNCE_RESTITUTION
            collided = True
        elif self.x > max_x:
            self.x = max_x
            if self.vx > 0:
                self.vx = -self.vx * WALL_BOUNCE_RESTITUTION
            collided = True

        # Top / Bottom walls
        if self.y < min_y:
            self.y = min_y
            if self.vy < 0:
                self.vy = -self.vy * WALL_BOUNCE_RESTITUTION
            collided = True
        elif self.y > max_y:
            self.y = max_y
            if self.vy > 0:
                self.vy = -self.vy * WALL_BOUNCE_RESTITUTION
            collided = True

        if collided and self.speed > 1.0:
            self.theta = math.atan2(self.vy, self.vx)

        return collided

    def handle_obstacle_collision(self, obstacle: StaticObstacle) -> bool:
        """
        Resolve collision with a static pillar: pushes robot out along normal
        and reflects velocity component.
        """
        collides, nx, ny, depth = obstacle.check_robot_collision(self.x, self.y, self.radius)
        if collides:
            # Positional displacement out of the obstacle
            self.x += nx * depth
            self.y += ny * depth

            # Velocity reflection along obstacle collision normal
            vel_normal = self.vx * nx + self.vy * ny
            if vel_normal < 0:
                # Robot is moving into the obstacle; reflect
                self.vx -= (1.0 + WALL_BOUNCE_RESTITUTION) * vel_normal * nx
                self.vy -= (1.0 + WALL_BOUNCE_RESTITUTION) * vel_normal * ny

            if self.speed > 1.0:
                self.theta = math.atan2(self.vy, self.vx)
            return True
        return False

    def reset(self, x: float, y: float, target_x: float, target_y: float) -> None:
        """Reset robot coordinates and destination."""
        self.x = float(x)
        self.y = float(y)
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.vx = 0.0
        self.vy = 0.0
        self.is_deflecting = False
        self.deflection_blend = 0.0
        self.deflection_angle_deg = 0.0
        self.deflection_threat_id = None
        self.trail.clear()
        self.trail.append((self.x, self.y))
        self._align_heading_to_target()


def resolve_robot_pairwise_collisions(robots: List[Robot]) -> None:
    """
    Physical contact resolution between robots.
    Prevents visual overlap by displacing overlapping robot centers along contact normal
    and exchanging momentum elastically.
    """
    n = len(robots)
    for i in range(n):
        for j in range(i + 1, n):
            r1 = robots[i]
            r2 = robots[j]

            dx = r2.x - r1.x
            dy = r2.y - r1.y
            dist_sq = dx * dx + dy * dy
            min_dist = r1.radius + r2.radius

            if dist_sq < min_dist * min_dist and dist_sq > 1e-9:
                dist = math.sqrt(dist_sq)
                overlap = (min_dist - dist)

                # Contact normal from r1 to r2
                nx = dx / dist
                ny = dy / dist

                # Positional displacement (equal mass share)
                half_overlap = overlap * 0.5
                r1.x -= nx * half_overlap
                r1.y -= ny * half_overlap
                r2.x += nx * half_overlap
                r2.y += ny * half_overlap

                # Relative velocity along collision normal
                rel_vx = r2.vx - r1.vx
                rel_vy = r2.vy - r1.vy
                vel_along_normal = rel_vx * nx + rel_vy * ny

                if vel_along_normal < 0:
                    # Robots moving toward each other; apply impulse
                    e = ROBOT_COLLISION_RESTITUTION
                    impulse = -(1.0 + e) * vel_along_normal * 0.5
                    r1.vx -= impulse * nx
                    r1.vy -= impulse * ny
                    r2.vx += impulse * nx
                    r2.vy += impulse * ny

                    # Update orientation
                    if r1.speed > 1.0:
                        r1.theta = math.atan2(r1.vy, r1.vx)
                    if r2.speed > 1.0:
                        r2.theta = math.atan2(r2.vy, r2.vx)


class ScenarioManager:
    """
    Orchestrates dynamic navigation scenarios designed to trigger all three risk tiers:
    - Safe (TTC > 3.0s)
    - Warning (1.0s < TTC <= 3.0s)
    - High Risk (TTC <= 1.0s / near-miss conflict)
    """

    SCENARIO_4WAY_CROSS = "4-Way Cross Intersection"
    SCENARIO_HEAD_ON = "Head-On Converge Conflict"
    SCENARIO_BOTTLENECK = "Chokepoint Bottleneck Transit"
    SCENARIO_CONTINUOUS_ROAM = "Continuous Dynamic Multi-Agent Roam"

    SCENARIOS = [
        SCENARIO_4WAY_CROSS,
        SCENARIO_HEAD_ON,
        SCENARIO_BOTTLENECK,
        SCENARIO_CONTINUOUS_ROAM,
    ]

    @classmethod
    def get_static_obstacles_for_scenario(cls, scenario_name: str) -> List[StaticObstacle]:
        """Generate static environmental obstacles tailored for the selected scenario."""
        cx = ARENA_X + ARENA_WIDTH / 2.0
        cy = ARENA_Y + ARENA_HEIGHT / 2.0

        if scenario_name == cls.SCENARIO_BOTTLENECK:
            # Two prominent vertical pillars leaving a central passage corridor
            gap = 140.0
            r = 45.0
            return [
                StaticObstacle(id=1, x=cx, y=cy - gap, radius=r, label="North Pillar"),
                StaticObstacle(id=2, x=cx, y=cy + gap, radius=r, label="South Pillar"),
            ]
        elif scenario_name == cls.SCENARIO_CONTINUOUS_ROAM:
            # Four symmetric pillars across the arena floor
            offset_x = ARENA_WIDTH * 0.24
            offset_y = ARENA_HEIGHT * 0.24
            r = 38.0
            return [
                StaticObstacle(id=1, x=cx - offset_x, y=cy - offset_y, radius=r, label="Pillar NW"),
                StaticObstacle(id=2, x=cx + offset_x, y=cy - offset_y, radius=r, label="Pillar NE"),
                StaticObstacle(id=3, x=cx - offset_x, y=cy + offset_y, radius=r, label="Pillar SW"),
                StaticObstacle(id=4, x=cx + offset_x, y=cy + offset_y, radius=r, label="Pillar SE"),
            ]
        elif scenario_name == cls.SCENARIO_HEAD_ON:
            # Two small side obstacles to guide traffic corridors
            r = 32.0
            return [
                StaticObstacle(id=1, x=cx, y=ARENA_Y + 120.0, radius=r, label="North Pylon"),
                StaticObstacle(id=2, x=cx, y=ARENA_Y + ARENA_HEIGHT - 120.0, radius=r, label="South Pylon"),
            ]
        else:
            # 4-Way Cross: Clear central intersection with corner guides
            r = 28.0
            return [
                StaticObstacle(id=1, x=ARENA_X + 110.0, y=ARENA_Y + 110.0, radius=r, label="Corner NW"),
                StaticObstacle(id=2, x=ARENA_X + ARENA_WIDTH - 110.0, y=ARENA_Y + 110.0, radius=r, label="Corner NE"),
                StaticObstacle(id=3, x=ARENA_X + 110.0, y=ARENA_Y + ARENA_HEIGHT - 110.0, radius=r, label="Corner SW"),
                StaticObstacle(id=4, x=ARENA_X + ARENA_WIDTH - 110.0, y=ARENA_Y + ARENA_HEIGHT - 110.0, radius=r, label="Corner SE"),
            ]

    @classmethod
    def setup_scenario(cls, scenario_name: str, num_robots: int = 5) -> Tuple[List[Robot], List[StaticObstacle]]:
        """
        Configure robot initial positions, target destinations, and routes for a given scenario.
        """
        cx = ARENA_X + ARENA_WIDTH / 2.0
        cy = ARENA_Y + ARENA_HEIGHT / 2.0
        w = ARENA_WIDTH
        h = ARENA_HEIGHT

        obstacles = cls.get_static_obstacles_for_scenario(scenario_name)
        robots: List[Robot] = []

        if scenario_name == cls.SCENARIO_4WAY_CROSS:
            # 4 robots start on the cardinal edges heading to opposite boundaries;
            # 5th robot navigates diagonally through the intersection.
            starts_and_targets = [
                # Robot 0: West -> East
                (ARENA_X + 60.0, cy, ARENA_X + w - 60.0, cy),
                # Robot 1: East -> West
                (ARENA_X + w - 60.0, cy + 10.0, ARENA_X + 60.0, cy + 10.0),
                # Robot 2: North -> South
                (cx, ARENA_Y + 60.0, cx, ARENA_Y + h - 60.0),
                # Robot 3: South -> North
                (cx + 10.0, ARENA_Y + h - 60.0, cx + 10.0, ARENA_Y + 60.0),
                # Robot 4: Diagonal NW -> SE (spawned clear of corner guide pillars)
                (ARENA_X + 160.0, ARENA_Y + 160.0, ARENA_X + w - 160.0, ARENA_Y + h - 160.0),
            ]
            for i in range(min(num_robots, len(starts_and_targets))):
                sx, sy, tx, ty = starts_and_targets[i]
                r = Robot(robot_id=i, x=sx, y=sy, target_x=tx, target_y=ty)
                # Assign ping-pong waypoint route
                r.set_waypoint_route([(tx, ty), (sx, sy)], loop=True)
                robots.append(r)

        elif scenario_name == cls.SCENARIO_HEAD_ON:
            # Pair 1: Head-on collision course along mid horizontal
            # Pair 2: Head-on collision course along mid vertical
            # Robot 4: Roaming cross-traffic
            starts_and_targets = [
                # Pair 1: Direct Head-on
                (ARENA_X + 70.0, cy - 40.0, ARENA_X + w - 70.0, cy - 40.0),
                (ARENA_X + w - 70.0, cy - 40.0, ARENA_X + 70.0, cy - 40.0),
                # Pair 2: Direct Head-on
                (cx - 50.0, ARENA_Y + 70.0, cx - 50.0, ARENA_Y + h - 70.0),
                (cx - 50.0, ARENA_Y + h - 70.0, cx - 50.0, ARENA_Y + 70.0),
                # Robot 4: Fast diagonal traverse
                (ARENA_X + 80.0, ARENA_Y + h - 80.0, ARENA_X + w - 80.0, ARENA_Y + 80.0),
            ]
            for i in range(min(num_robots, len(starts_and_targets))):
                sx, sy, tx, ty = starts_and_targets[i]
                r = Robot(robot_id=i, x=sx, y=sy, target_x=tx, target_y=ty)
                r.set_waypoint_route([(tx, ty), (sx, sy)], loop=True)
                robots.append(r)

        elif scenario_name == cls.SCENARIO_BOTTLENECK:
            # Robots converge from both left & right halves through the central chokepoint
            starts_and_targets = [
                # 3 robots on West side heading to East side through bottleneck
                (ARENA_X + 90.0, cy - 120.0, ARENA_X + w - 90.0, cy + 120.0),
                (ARENA_X + 80.0, cy, ARENA_X + w - 80.0, cy),
                (ARENA_X + 90.0, cy + 120.0, ARENA_X + w - 90.0, cy - 120.0),
                # 2 robots on East side heading to West side
                (ARENA_X + w - 90.0, cy - 60.0, ARENA_X + 90.0, cy + 60.0),
                (ARENA_X + w - 80.0, cy + 60.0, ARENA_X + 90.0, cy - 60.0),
            ]
            for i in range(min(num_robots, len(starts_and_targets))):
                sx, sy, tx, ty = starts_and_targets[i]
                r = Robot(robot_id=i, x=sx, y=sy, target_x=tx, target_y=ty)
                # Waypoints route through center (cx, cy)
                route = [(cx, cy), (tx, ty), (cx, cy), (sx, sy)]
                r.set_waypoint_route(route, loop=True)
                robots.append(r)

        else:  # SCENARIO_CONTINUOUS_ROAM
            # Randomized initial positions and multi-waypoint patrol loops
            margin = 90.0
            for i in range(num_robots):
                # Distribute initial spawns
                angle = (2.0 * math.pi / num_robots) * i
                spawn_r = min(w, h) * 0.35
                sx = cx + math.cos(angle) * spawn_r
                sy = cy + math.sin(angle) * spawn_r

                # Multi-point loop
                num_wp = 4
                wps = []
                for k in range(num_wp):
                    wp_angle = angle + (2.0 * math.pi / num_wp) * (k + 1)
                    wpx = cx + math.cos(wp_angle) * (spawn_r * random.uniform(0.7, 1.2))
                    wpy = cy + math.sin(wp_angle) * (spawn_r * random.uniform(0.7, 1.2))
                    # Clamp inside arena
                    wpx = max(ARENA_X + margin, min(ARENA_X + w - margin, wpx))
                    wpy = max(ARENA_Y + margin, min(ARENA_Y + h - margin, wpy))
                    wps.append((wpx, wpy))

                r = Robot(robot_id=i, x=sx, y=sy, target_x=wps[0][0], target_y=wps[0][1])
                r.set_waypoint_route(wps, loop=True)
                robots.append(r)

        return robots, obstacles

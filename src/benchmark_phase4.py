"""
Headless Reproducible Benchmark Suite for Phase 4 (Stages E–G).

Compares 500 Baseline Episodes vs. 500 Adaptive Episodes under strictly paired initial conditions.
Evaluates:
- Collision Rate & Total Discrete Physical Collisions
- Navigation Success Rate (Goal reached with zero collisions)
- Timeout Rate
- Total & Per-Robot Travel Path Length
- Near-Miss Conflict Frequency
- GNN Online Inference & Deflection Latency (mean, p50, p95, p99)
- Measured Simulation Throughput (FPS)
"""

import os
import sys
import time
import math
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set dummy video driver for pure headless simulation
os.environ["SDL_VIDEODRIVER"] = "dummy"

from config import (
    ARENA_X,
    ARENA_Y,
    ARENA_WIDTH,
    ARENA_HEIGHT,
    ROBOT_RADIUS,
    ROBOT_MAX_SPEED,
    GOAL_REACH_DISTANCE,
    RISK_LABEL_HIGH_RISK,
)
from robot_state import (
    Robot,
    StaticObstacle,
    ScenarioManager,
    resolve_robot_pairwise_collisions,
)
from src.gnn_inference_controller import GNNInferenceController
from src.adaptive_deflection import AdaptiveDeflectionController


@dataclass
class EpisodeResult:
    """Detailed performance metrics for a single simulation episode."""
    episode_id: int
    mode: str                     # "baseline" or "adaptive"
    seed: int
    scenario: str
    num_robots: int
    success: bool                 # Reached destination without any collisions
    had_collision: bool           # Experienced at least one physical collision
    collision_count: int          # Total discrete collision events
    timeout: bool                 # Reached max steps before completion
    steps: int
    sim_time_s: float
    wall_time_s: float
    total_path_length: float      # Combined Euclidean distance traveled by all robots
    avg_robot_path: float         # Mean path distance per robot
    near_miss_count: int          # Encounters entering close conflict [36px, 80px]
    min_robot_dist: float         # Minimum distance recorded between any robot pair
    mean_latency_ms: float        # AI inference time per step (ms)
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    fps: float                    # Simulation throughput (steps / wall second)


class Phase4BenchmarkRunner:
    """
    Orchestrates headless comparative benchmarking between baseline and adaptive modes.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/gnn_interaction_model.pth",
        scalers_path: str = "checkpoints/gnn_scalers.pkl",
        max_steps_per_episode: int = 520,
        fps: int = 60,
        inference_interval: int = 2,
    ):
        self.max_steps = max_steps_per_episode
        self.dt = 1.0 / float(fps)
        self.fps = fps
        self.inference_interval = inference_interval

        # Initialize AI Controllers (CPU optimal for low-latency micro-graph inference)
        self.gnn_controller = GNNInferenceController(
            checkpoint_path=checkpoint_path,
            scalers_path=scalers_path,
            inference_interval=inference_interval,
            device=torch.device("cpu"),
        )
        self.adaptive_controller = AdaptiveDeflectionController()

    def generate_paired_episode_setup(
        self,
        seed: int,
        scenario_name: str,
        num_robots: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[StaticObstacle]]:
        """
        Generates identical, reproducible initial conditions for a paired episode.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Base scenario placement
        robots, obstacles = ScenarioManager.setup_scenario(scenario_name, num_robots=num_robots)

        # Apply deterministic perturbation based on seed
        saved_robot_states: List[Dict[str, Any]] = []
        for r in robots:
            # Subtle perturbation to create varied conflict trajectories
            dx_perturb = random.uniform(-18.0, 18.0)
            dy_perturb = random.uniform(-18.0, 18.0)
            r.x = max(ARENA_X + r.radius + 10.0, min(ARENA_X + ARENA_WIDTH - r.radius - 10.0, r.x + dx_perturb))
            r.y = max(ARENA_Y + r.radius + 10.0, min(ARENA_Y + ARENA_HEIGHT - r.radius - 10.0, r.y + dy_perturb))

            tx_perturb = random.uniform(-20.0, 20.0)
            ty_perturb = random.uniform(-20.0, 20.0)
            r.target_x = max(ARENA_X + 40.0, min(ARENA_X + ARENA_WIDTH - 40.0, r.target_x + tx_perturb))
            r.target_y = max(ARENA_Y + 40.0, min(ARENA_Y + ARENA_HEIGHT - 40.0, r.target_y + ty_perturb))

            r._align_heading_to_target()

            saved_robot_states.append({
                "id": r.id,
                "x": r.x,
                "y": r.y,
                "target_x": r.target_x,
                "target_y": r.target_y,
                "vx": r.vx,
                "vy": r.vy,
                "theta": r.theta,
                "radius": r.radius,
                "max_speed": r.max_speed,
                "color": r.color,
            })

        return saved_robot_states, obstacles

    def _instantiate_robots_from_states(self, states: List[Dict[str, Any]]) -> List[Robot]:
        """Reconstructs identical Robot objects from saved state dictionary."""
        robots = []
        for s in states:
            r = Robot(
                robot_id=s["id"],
                x=s["x"],
                y=s["y"],
                target_x=s["target_x"],
                target_y=s["target_y"],
                color=s["color"],
                radius=s["radius"],
                max_speed=s["max_speed"],
            )
            r.vx = s["vx"]
            r.vy = s["vy"]
            r.theta = s["theta"]
            robots.append(r)
        return robots

    def run_episode(
        self,
        episode_id: int,
        mode: str,
        seed: int,
        scenario_name: str,
        initial_states: List[Dict[str, Any]],
        obstacles: List[StaticObstacle],
    ) -> EpisodeResult:
        """
        Executes a single episode in either 'baseline' or 'adaptive' mode.
        """
        robots = self._instantiate_robots_from_states(initial_states)
        num_robots = len(robots)
        is_adaptive = (mode == "adaptive")

        if is_adaptive:
            self.adaptive_controller.reset_profiles()

        # Telemetry & Metrics Tracking
        total_collisions = 0
        active_colliding_pairs = set()
        active_conflicts = {}
        near_miss_count = 0
        min_distance_overall = float("inf")

        prev_positions = {r.id: (r.x, r.y) for r in robots}
        robot_path_distances = {r.id: 0.0 for r in robots}

        latency_measurements_ms: List[float] = []

        step = 0
        sim_time = 0.0
        wall_start = time.perf_counter()

        target_destinations = {s["id"]: (s["target_x"], s["target_y"]) for s in initial_states}
        reached_goal = {r.id: False for r in robots}

        while step < self.max_steps:
            # 1. AI Inference & Steering
            if is_adaptive:
                t_ai_start = time.perf_counter()
                gnn_result = self.gnn_controller.predict_frame(
                    robots=robots, frame_id=step, sim_time=sim_time
                )
                self.gnn_controller.update_robot_risk_states(robots, gnn_result)
                self.adaptive_controller.apply_deflection(
                    robots=robots, obstacles=obstacles, gnn_result=gnn_result, dt=self.dt
                )
                t_ai_end = time.perf_counter()
                latency_measurements_ms.append((t_ai_end - t_ai_start) * 1000.0)
            else:
                for r in robots:
                    r.update_kinematics(self.dt)
                    r.is_deflecting = False
                    r.deflection_blend = 0.0

            # 2. Environmental Boundaries & Obstacles
            for r in robots:
                r.handle_boundary_collision()
                for obs in obstacles:
                    r.handle_obstacle_collision(obs)

            # 3. Path Length Accumulation
            for r in robots:
                px, py = prev_positions[r.id]
                dist_moved = math.hypot(r.x - px, r.y - py)
                robot_path_distances[r.id] += dist_moved
                prev_positions[r.id] = (r.x, r.y)

            # 4. Pairwise Contact & Conflict Analysis
            current_colliding_pairs = set()
            for i in range(num_robots):
                for j in range(i + 1, num_robots):
                    r_a = robots[i]
                    r_b = robots[j]
                    d_ab = math.hypot(r_b.x - r_a.x, r_b.y - r_a.y)
                    min_contact = r_a.radius + r_b.radius
                    pair_key = (min(r_a.id, r_b.id), max(r_a.id, r_b.id))

                    if d_ab < min_distance_overall:
                        min_distance_overall = d_ab

                    # Physical Collision
                    if d_ab <= min_contact:
                        current_colliding_pairs.add(pair_key)
                        if pair_key not in active_colliding_pairs:
                            total_collisions += 1

                    # Near-Miss Conflict Detection [contact threshold, 80 px]
                    elif min_contact < d_ab <= 80.0:
                        # Closing velocity check
                        dx = r_b.x - r_a.x
                        dy = r_b.y - r_a.y
                        dvx = r_b.vx - r_a.vx
                        dvy = r_b.vy - r_a.vy
                        closing_spd = -(dx * dvx + dy * dvy) / max(1e-6, d_ab)
                        if closing_spd > 5.0 and pair_key not in active_conflicts:
                            near_miss_count += 1
                            active_conflicts[pair_key] = True

                    # Remove conflict when cleared
                    if d_ab > 100.0 and pair_key in active_conflicts:
                        del active_conflicts[pair_key]

            active_colliding_pairs = current_colliding_pairs

            # 5. Inter-Robot Contact Physics Resolution
            resolve_robot_pairwise_collisions(robots)

            step += 1
            sim_time += self.dt

            # 6. Check Primary Goal Arrival for All Robots
            for r in robots:
                if not reached_goal[r.id]:
                    tx, ty = target_destinations[r.id]
                    if math.hypot(r.x - tx, r.y - ty) <= GOAL_REACH_DISTANCE:
                        reached_goal[r.id] = True

            if all(reached_goal.values()):
                break

        wall_duration = time.perf_counter() - wall_start
        fps_measured = step / max(1e-6, wall_duration)

        total_path = sum(robot_path_distances.values())
        avg_path = total_path / max(1, num_robots)

        all_reached = all(reached_goal.values())
        had_collision = (total_collisions > 0)
        # Success defined strictly: all robots reached target with zero collisions
        success = (all_reached and not had_collision)
        timeout = (step >= self.max_steps and not all_reached)

        # Latency statistics
        if latency_measurements_ms:
            mean_lat = float(np.mean(latency_measurements_ms))
            p50_lat = float(np.percentile(latency_measurements_ms, 50))
            p95_lat = float(np.percentile(latency_measurements_ms, 95))
            p99_lat = float(np.percentile(latency_measurements_ms, 99))
        else:
            mean_lat, p50_lat, p95_lat, p99_lat = 0.0, 0.0, 0.0, 0.0

        return EpisodeResult(
            episode_id=episode_id,
            mode=mode,
            seed=seed,
            scenario=scenario_name,
            num_robots=num_robots,
            success=success,
            had_collision=had_collision,
            collision_count=total_collisions,
            timeout=timeout,
            steps=step,
            sim_time_s=sim_time,
            wall_time_s=wall_duration,
            total_path_length=total_path,
            avg_robot_path=avg_path,
            near_miss_count=near_miss_count,
            min_robot_dist=min_distance_overall,
            mean_latency_ms=mean_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            fps=fps_measured,
        )

    def run_paired_benchmark(
        self,
        num_episodes: int = 500,
        output_csv: str = "results/phase4_benchmark.csv",
        log_interval: int = 50,
    ) -> pd.DataFrame:
        """
        Executes paired episodes across all scenarios and exports comprehensive CSV.
        """
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results: List[EpisodeResult] = []

        scenarios = ScenarioManager.SCENARIOS

        print("=" * 76)
        print(f" STARTING PHASE 4 BENCHMARK: {num_episodes} PAIRED EPISODES (1,000 RUNS)")
        print(f" Scenarios: {len(scenarios)} Distinct Topologies")
        print(f" Target CSV: {output_csv}")
        print("=" * 76)

        t_bench_start = time.perf_counter()

        for ep_idx in range(num_episodes):
            seed = ep_idx
            scenario_name = scenarios[ep_idx % len(scenarios)]

            # Generate identical setup for this seed
            initial_states, obstacles = self.generate_paired_episode_setup(
                seed=seed, scenario_name=scenario_name, num_robots=5
            )

            # 1. Baseline Episode Run
            res_baseline = self.run_episode(
                episode_id=ep_idx,
                mode="baseline",
                seed=seed,
                scenario_name=scenario_name,
                initial_states=initial_states,
                obstacles=obstacles,
            )
            results.append(res_baseline)

            # 2. Adaptive Episode Run (Paired with identical initial states)
            res_adaptive = self.run_episode(
                episode_id=ep_idx,
                mode="adaptive",
                seed=seed,
                scenario_name=scenario_name,
                initial_states=initial_states,
                obstacles=obstacles,
            )
            results.append(res_adaptive)

            if (ep_idx + 1) % log_interval == 0 or (ep_idx + 1) == num_episodes:
                elapsed = time.perf_counter() - t_bench_start
                # Compute running comparison
                df_curr = pd.DataFrame([asdict(r) for r in results])
                b_curr = df_curr[df_curr["mode"] == "baseline"]
                a_curr = df_curr[df_curr["mode"] == "adaptive"]

                b_succ = (b_curr["success"].sum() / len(b_curr)) * 100.0
                a_succ = (a_curr["success"].sum() / len(a_curr)) * 100.0
                b_coll = b_curr["had_collision"].sum()
                a_coll = a_curr["had_collision"].sum()

                print(
                    f" [{ep_idx + 1:3d}/{num_episodes}] Elapsed: {elapsed:5.1f}s | "
                    f"Baseline Succ: {b_succ:5.1f}% (Coll: {b_coll:3d}) | "
                    f"Adaptive Succ: {a_succ:5.1f}% (Coll: {a_coll:3d}) | "
                    f"Mean Lat: {a_curr['mean_latency_ms'].mean():.2f}ms"
                )

        total_elapsed = time.perf_counter() - t_bench_start
        print("=" * 76)
        print(f" BENCHMARK COMPLETE in {total_elapsed:.1f}s ({total_elapsed/60:.2f} mins)")
        print("=" * 76)

        df_final = pd.DataFrame([asdict(r) for r in results])
        df_final.to_csv(output_csv, index=False)
        print(f" Successfully exported {len(df_final)} rows to {output_csv}")
        return df_final


def measure_pure_simulator_fps(num_steps: int = 1000) -> float:
    """
    Measures raw simulation throughput with all AI controllers disabled.
    """
    robots, obstacles = ScenarioManager.setup_scenario(ScenarioManager.SCENARIO_4WAY_CROSS, num_robots=5)
    dt = 1.0 / 60.0

    t_start = time.perf_counter()
    for _ in range(num_steps):
        for r in robots:
            r.update_kinematics(dt)
            r.advance_waypoint_if_reached()
            r.handle_boundary_collision()
            for obs in obstacles:
                r.handle_obstacle_collision(obs)
        resolve_robot_pairwise_collisions(robots)

    duration = time.perf_counter() - t_start
    fps = num_steps / max(1e-6, duration)
    return fps


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500, help="Number of paired episodes to run")
    parser.add_argument("--smoke", action="store_true", help="Run quick 5-episode smoke benchmark")
    parser.add_argument("--output", type=str, default="results/phase4_benchmark.csv", help="Output CSV path")
    args = parser.parse_args()

    num_episodes = 5 if args.smoke else args.episodes
    runner = Phase4BenchmarkRunner()

    # Verify original simulator speed
    raw_fps = measure_pure_simulator_fps(1000)
    print(f" Verified Raw Baseline Simulator Speed: {raw_fps:,.0f} FPS")

    runner.run_paired_benchmark(num_episodes=num_episodes, output_csv=args.output)

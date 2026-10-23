"""
Main Entry Point: Intelligent Collision-Risk Prediction For Multi-Robot Navigation.
Phase 1: Advanced UI & Data Generation Engine.

Integrates:
- Visual Frontend (ui_engine.py)
- Kinematics & Physics Backend (robot_state.py)
- Analytical TTC & Dataset Extraction (data_collector.py)

Usage:
  python3 main.py                       # Launch interactive visualizer (Scenario 1, 5 robots)
  python3 main.py --scenario 2          # Launch with Head-On Conflict scenario
  python3 main.py --robots 4            # Launch with 4 robots
  python3 main.py --headless --steps 1000 # Headless fast-mode to generate 1000 steps of CSV data
"""

import os
import sys
import time
import math
import argparse
import pygame
import torch
from typing import List

from config import (
    FPS,
    DEFAULT_ROBOT_COUNT,
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_FILENAME,
    DATA_LOG_INTERVAL_FRAMES,
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
from data_collector import DataCollector
from ui_engine import UIEngine
from src.gnn_inference_controller import GNNInferenceController
from src.adaptive_deflection import AdaptiveDeflectionController


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: Intelligent Collision-Risk Prediction & Adaptive Deflection Engine"
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Initial scenario: 1=4-Way Cross, 2=Head-On Conflict, 3=Bottleneck Transit, 4=Continuous Roam (Default: 1)",
    )
    parser.add_argument(
        "--robots",
        type=int,
        default=DEFAULT_ROBOT_COUNT,
        help=f"Number of autonomous robots to simulate (Default: {DEFAULT_ROBOT_COUNT})",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable automatic CSV recording on startup (can be toggled with 'R')",
    )
    parser.add_argument(
        "--no-deflection",
        action="store_true",
        help="Start with GNN adaptive deflection disabled (can be toggled in real-time with 'D')",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=DEFAULT_DATASET_FILENAME,
        help=f"Filename for CSV dataset output (Default: {DEFAULT_DATASET_FILENAME})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run simulation headlessly at maximum speed without opening a visual window",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Maximum simulation frames to run (0 = run indefinitely until quit)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help=f"Target frame rate for simulation (Default: {FPS})",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine scenario name
    scenario_map = {
        1: ScenarioManager.SCENARIO_4WAY_CROSS,
        2: ScenarioManager.SCENARIO_HEAD_ON,
        3: ScenarioManager.SCENARIO_BOTTLENECK,
        4: ScenarioManager.SCENARIO_CONTINUOUS_ROAM,
    }
    current_scenario_idx = args.scenario
    current_scenario_name = scenario_map[current_scenario_idx]

    # Handle Headless Mode setup if requested
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Initialize Pygame and Subsystems
    pygame.init()
    clock = pygame.time.Clock()

    # Instantiate Data Collector
    data_collector = DataCollector(
        dataset_dir=DEFAULT_DATASET_DIR,
        filename=args.dataset_file,
        auto_record=not args.no_record,
    )

    # Initialize Robots & Static Obstacles for chosen scenario
    robots, obstacles = ScenarioManager.setup_scenario(
        current_scenario_name, num_robots=args.robots
    )

    # Initialize Visual Frontend (only if not headless)
    ui_engine = None if args.headless else UIEngine()

    # Initialize Phase 4 AI Controllers
    gnn_controller = None
    adaptive_deflection = None
    try:
        gnn_controller = GNNInferenceController(
            checkpoint_path="checkpoints/gnn_interaction_model.pth",
            scalers_path="checkpoints/gnn_scalers.pkl",
            inference_interval=1,
            device=torch.device("cpu"),
        )
        adaptive_deflection = AdaptiveDeflectionController()
        print(" [AI CONTROLLER] GNN Inference & Adaptive Deflection Engines Ready.")
    except Exception as e:
        print(f" [AI CONTROLLER] Note: GNN/Deflection running in fallback mode: {e}")

    deflection_enabled = not args.no_deflection
    total_physical_collisions = 0
    total_avoidance_events = 0
    active_colliding_pairs = set()
    active_conflicts = {}

    print("=" * 72)
    print("   INTELLIGENT COLLISION-RISK PREDICTION: MULTI-ROBOT NAVIGATION")
    print("   Phase 4: Real-Time GNN Inference & Adaptive Deflection Engine")
    print("=" * 72)
    print(f" Active Scenario:    {current_scenario_name}")
    print(f" Robot Fleet:        {len(robots)} Active Autonomous Agents")
    print(f" Static Obstacles:   {len(obstacles)} Pillars Loaded")
    print(f" AI Deflection:      {'ENABLED' if deflection_enabled else 'DISABLED'}")
    print(f" CSV Dataset Target: {data_collector.filepath}")
    print(f" Auto-Recording:     {'ENABLED' if data_collector.is_recording else 'DISABLED'}")
    print(f" Simulation Mode:    {'HEADLESS' if args.headless else 'INTERACTIVE PYGAME'}")
    print("-" * 72)
    print(" Controls: [SPACE] Pause | [D] Deflection Toggle | [R] REC | [1-4] Scenario | [ESC] Quit")
    print("=" * 72)

    # Simulation State Tracking
    frame_id = 0
    sim_time = 0.0
    dt = 1.0 / float(args.fps)
    is_paused = False
    running = True

    try:
        while running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        is_paused = not is_paused
                        print(f" [SIMULATION] {'PAUSED' if is_paused else 'RESUMED'}")

                    elif event.key == pygame.K_d:
                        deflection_enabled = not deflection_enabled
                        if not deflection_enabled and adaptive_deflection:
                            adaptive_deflection.reset_profiles()
                            for r in robots:
                                r.is_deflecting = False
                                r.deflection_blend = 0.0
                        print(f" [AI DEFLECTION] {'ENABLED (Active Evasion)' if deflection_enabled else 'DISABLED (Baseline Mode)'}")

                    elif event.key == pygame.K_r:
                        rec = data_collector.toggle_recording()
                        print(f" [DATA RECORDER] {'RECORDING RESUMED' if rec else 'RECORDING PAUSED'}")

                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                        key_val = event.key - pygame.K_0
                        current_scenario_idx = key_val
                        current_scenario_name = scenario_map[current_scenario_idx]
                        robots, obstacles = ScenarioManager.setup_scenario(
                            current_scenario_name, num_robots=args.robots
                        )
                        active_colliding_pairs.clear()
                        active_conflicts.clear()
                        if adaptive_deflection:
                            adaptive_deflection.reset_profiles()
                        print(f" [SCENARIO SWITCH] -> {current_scenario_name}")

                    elif event.key == pygame.K_n:
                        # Regenerate / shuffle destinations for all robots
                        for r in robots:
                            r.advance_waypoint_if_reached()
                        print(" [NAVIGATION] Targets refreshed.")

                    elif event.key == pygame.K_c:
                        data_collector.flush()
                        st = data_collector.get_stats()
                        print(f" [CSV STATS] Logged {st['total_records']} rows | "
                              f"Safe: {st['safe_count']} ({st['safe_pct']:.1f}%) | "
                              f"Warning: {st['warning_count']} ({st['warning_pct']:.1f}%) | "
                              f"High Risk: {st['high_risk_count']} ({st['high_risk_pct']:.1f}%)")

            # 2. GNN Inference & Adaptive Deflection Steering
            gnn_result = None
            if not is_paused:
                if gnn_controller is not None:
                    try:
                        gnn_result = gnn_controller.predict_frame(
                            robots=robots, frame_id=frame_id, sim_time=sim_time
                        )
                        gnn_controller.update_robot_risk_states(robots, gnn_result)
                    except Exception:
                        gnn_result = None

                if deflection_enabled and adaptive_deflection is not None:
                    adaptive_deflection.apply_deflection(
                        robots=robots, obstacles=obstacles, gnn_result=gnn_result, dt=dt
                    )
                else:
                    for robot in robots:
                        robot.update_kinematics(dt)
                        robot.is_deflecting = False
                        robot.deflection_blend = 0.0

                # Check destination arrivals & enforce environmental boundaries
                for robot in robots:
                    robot.advance_waypoint_if_reached()
                    robot.handle_boundary_collision()
                    for obs in obstacles:
                        robot.handle_obstacle_collision(obs)

                # Reliable Collision & Avoidance Measurement
                current_colliding_pairs = set()
                n_robots = len(robots)
                for i in range(n_robots):
                    for j in range(i + 1, n_robots):
                        r_a, r_b = robots[i], robots[j]
                        dist_ab = math.hypot(r_b.x - r_a.x, r_b.y - r_a.y)
                        pair_key = (min(r_a.id, r_b.id), max(r_a.id, r_b.id))

                        # 1. Actual physical contact collision
                        if dist_ab <= (r_a.radius + r_b.radius):
                            current_colliding_pairs.add(pair_key)
                            if pair_key not in active_colliding_pairs:
                                total_physical_collisions += 1
                                if pair_key in active_conflicts:
                                    active_conflicts[pair_key]["collided"] = True

                        # 2. Reliable avoidance event tracking
                        is_threatened = (
                            r_a.risk_label == RISK_LABEL_HIGH_RISK
                            or r_b.risk_label == RISK_LABEL_HIGH_RISK
                            or dist_ab < 100.0
                        )
                        is_evading = (
                            r_a.is_deflecting
                            or r_b.is_deflecting
                            or getattr(r_a, "deflection_blend", 0.0) > 0.05
                            or getattr(r_b, "deflection_blend", 0.0) > 0.05
                        )

                        if pair_key not in active_conflicts:
                            if is_threatened and dist_ab < 120.0:
                                active_conflicts[pair_key] = {
                                    "min_dist": dist_ab,
                                    "deflected": is_evading,
                                    "collided": False,
                                }
                        else:
                            rec = active_conflicts[pair_key]
                            if is_evading:
                                rec["deflected"] = True
                            rec["min_dist"] = min(rec["min_dist"], dist_ab)

                            # Check for safe divergence
                            dx_rel = r_b.x - r_a.x
                            dy_rel = r_b.y - r_a.y
                            dvx_rel = r_b.vx - r_a.vx
                            dvy_rel = r_b.vy - r_a.vy
                            closing_spd = -(dx_rel * dvx_rel + dy_rel * dvy_rel) / max(1e-6, dist_ab)

                            if dist_ab > 125.0 and closing_spd <= 0.0:
                                if rec["deflected"] and not rec["collided"] and rec["min_dist"] < 95.0:
                                    total_avoidance_events += 1
                                del active_conflicts[pair_key]

                active_colliding_pairs = current_colliding_pairs

                # Inter-robot physical contact resolution (Preserved safety backstop)
                resolve_robot_pairwise_collisions(robots)

                sim_time += dt
                frame_id += 1

            # 3. Analytical TTC & Data Extraction
            # Log on every designated frame interval (or every frame)
            should_log = (frame_id % DATA_LOG_INTERVAL_FRAMES == 0) and not is_paused
            pairwise_summaries = data_collector.process_and_log_frame(
                frame_id=frame_id,
                timestamp=sim_time,
                scenario_name=current_scenario_name,
                robots=robots,
            )

            # 4. Visual Rendering
            if ui_engine:
                stats = data_collector.get_stats()
                ui_engine.render(
                    sim_time=sim_time,
                    fps=clock.get_fps() if not is_paused else 0.0,
                    scenario_name=current_scenario_name,
                    robots=robots,
                    obstacles=obstacles,
                    pairwise_summaries=pairwise_summaries,
                    stats=stats,
                    is_paused=is_paused,
                    deflection_enabled=deflection_enabled,
                    total_collisions=total_physical_collisions,
                    total_avoidances=total_avoidance_events,
                    gnn_result=gnn_result,
                )

            # Step limit check for batch data generation
            if args.steps > 0 and frame_id >= args.steps:
                print(f" Reached target limit of {args.steps} simulation steps.")
                running = False

            if not args.headless:
                clock.tick(args.fps)

    except KeyboardInterrupt:
        print("\n [SIMULATION] Interrupted by user.")

    finally:
        # Graceful cleanup and dataset summary
        data_collector.close()
        pygame.quit()

        st = data_collector.get_stats()
        print("\n" + "=" * 72)
        print("   SIMULATION SESSION TERMINATED - DATASET SUMMARY")
        print("=" * 72)
        print(f" Dataset Location:      {st['filepath']}")
        print(f" Total Pairwise Rows:   {st['total_records']:,}")
        print(f"   - Safe (TTC > 3s):    {st['safe_count']:,} ({st['safe_pct']:.2f}%)")
        print(f"   - Warning (1-3s):     {st['warning_count']:,} ({st['warning_pct']:.2f}%)")
        print(f"   - High Risk (<= 1s):  {st['high_risk_count']:,} ({st['high_risk_pct']:.2f}%)")
        print("=" * 72)
        print(" Phase 1 Execution Complete. Ready for Phase 2 (Baseline Predictive Engine).")


if __name__ == "__main__":
    main()

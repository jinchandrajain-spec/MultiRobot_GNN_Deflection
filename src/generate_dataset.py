"""
Dataset Generation Utility for Multi-Robot Navigation Simulation.
Generates comprehensive multi-scenario dataset covering all 4 benchmark scenarios:
1. 4-Way Cross Intersection
2. Head-On Converge Conflict
3. Chokepoint Bottleneck Transit
4. Continuous Dynamic Multi-Agent Roam

Ensures balanced representation of Safe (0), Warning (1), and High Risk (2) states.
"""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DEFAULT_DATASET_DIR, DEFAULT_DATASET_FILENAME
from robot_state import ScenarioManager, resolve_robot_pairwise_collisions
from data_collector import DataCollector


def generate_dataset(
    frames_per_scenario: int = 1200,
    num_robots: int = 5,
    output_dir: str = DEFAULT_DATASET_DIR,
    output_file: str = DEFAULT_DATASET_FILENAME,
) -> str:
    """
    Simulates each scenario headlessly for `frames_per_scenario` steps and logs
    all pairwise interaction records to a single CSV file.
    """
    filepath = os.path.join(output_dir, output_file)
    if os.path.exists(filepath):
        os.remove(filepath)

    collector = DataCollector(dataset_dir=output_dir, filename=output_file, auto_record=True)
    dt = 1.0 / 60.0
    global_frame = 0

    print("=" * 70)
    print("   GENERATING MULTI-SCENARIO COLLISION-RISK DATASET")
    print("=" * 70)

    for scen_idx, scenario_name in enumerate(ScenarioManager.SCENARIOS, 1):
        print(f" Simulating Scenario {scen_idx}/{len(ScenarioManager.SCENARIOS)}: {scenario_name}...")
        robots, obstacles = ScenarioManager.setup_scenario(scenario_name, num_robots=num_robots)

        for step in range(frames_per_scenario):
            global_frame += 1
            sim_time = global_frame * dt

            for r in robots:
                r.update_kinematics(dt)
                r.advance_waypoint_if_reached()
                r.handle_boundary_collision()
                for obs in obstacles:
                    r.handle_obstacle_collision(obs)

            resolve_robot_pairwise_collisions(robots)

            collector.process_and_log_frame(
                frame_id=global_frame,
                timestamp=sim_time,
                scenario_name=scenario_name,
                robots=robots,
            )

        st = collector.get_stats()
        print(f"   -> Cumulative Rows: {st['total_records']:,} | "
              f"Safe: {st['safe_pct']:.1f}% | Warn: {st['warning_pct']:.1f}% | High: {st['high_risk_pct']:.1f}%")

    collector.close()

    final_stats = collector.get_stats()
    print("=" * 70)
    print(f" Dataset Generation Complete: {collector.filepath}")
    print(f" Total Rows:      {final_stats['total_records']:,}")
    print(f" Safe (0):        {final_stats['safe_count']:,} ({final_stats['safe_pct']:.2f}%)")
    print(f" Warning (1):     {final_stats['warning_count']:,} ({final_stats['warning_pct']:.2f}%)")
    print(f" High Risk (2):   {final_stats['high_risk_count']:,} ({final_stats['high_risk_pct']:.2f}%)")
    print("=" * 70)
    return collector.filepath


if __name__ == "__main__":
    generate_dataset(frames_per_scenario=1200, num_robots=5)

"""
Data Extraction & Analytical Time-To-Collision (TTC) Engine.
Phase 1: Advanced UI & Data Generation Engine.

Responsibilities:
- Mathematically rigorous analytical TTC computation for 2D kinematic vectors.
- Pairwise feature extraction across all robot combinations.
- Risk classification labeling (0: Safe, 1: Warning, 2: High Risk).
- High-throughput CSV dataset writer with buffering, stats, and Pandas export.
"""

import os
import math
import csv
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd

from config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_FILENAME,
    TTC_SAFE_THRESHOLD,
    TTC_WARNING_THRESHOLD,
    TTC_MAX_CLIP,
    RISK_LABEL_SAFE,
    RISK_LABEL_WARNING,
    RISK_LABEL_HIGH_RISK,
    RISK_NAMES,
    AUTO_FLUSH_INTERVAL,
)
from robot_state import Robot


def compute_analytical_ttc(
    x1: float,
    y1: float,
    vx1: float,
    vy1: float,
    r1: float,
    x2: float,
    y2: float,
    vx2: float,
    vy2: float,
    r2: float,
    safety_margin: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Computes exact closed-form Time-To-Collision (TTC) between two circular agents
    moving with linear velocities.

    Mathematical Formulation:
      Relative position:  dp = p2 - p1 = (x2 - x1, y2 - y1)
      Relative velocity:  dv = v2 - v1 = (vx2 - vx1, vy2 - vy1)
      Combined radius:    R = r1 + r2 + safety_margin

      Distance squared over time:
        ||d(t)||^2 = ||dp + dv * t||^2 = ||dv||^2 * t^2 + 2 * (dp . dv) * t + ||dp||^2

      Collision condition:
        ||d(t)||^2 <= R^2  <=>  a * t^2 + b * t + c <= 0
      where:
        a = ||dv||^2
        b = 2 * (dp . dv)
        c = ||dp||^2 - R^2

    Returns:
      (ttc, closest_approach_distance, closing_speed)
        - ttc: Time to collision in seconds, or float('inf') if trajectories do not collide.
        - closest_approach_distance: Projected minimum center distance along current velocity rays.
        - closing_speed: Rate at which agents are closing distance (-dp . dv / ||dp||).
    """
    dx = x2 - x1
    dy = y2 - y1
    dist_sq = dx * dx + dy * dy
    dist = math.sqrt(dist_sq)

    dvx = vx2 - vx1
    dvy = vy2 - vy1
    combined_r = r1 + r2 + safety_margin

    # Closing speed: positive when approaching, negative when moving apart
    if dist > 1e-6:
        closing_speed = -(dx * dvx + dy * dvy) / dist
    else:
        closing_speed = 0.0

    # Case 1: Already in physical contact or overlapping
    if dist <= combined_r:
        return 0.0, dist, closing_speed

    # Coefficients of quadratic: a * t^2 + b * t + c = 0
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (dx * dvx + dy * dvy)
    c = dist_sq - (combined_r * combined_r)

    # Case 2: Relative velocity is virtually zero (moving identically or stationary)
    if a < 1e-8:
        return float("inf"), dist, closing_speed

    # Time to closest point of approach (CPA)
    t_cpa = max(0.0, -b / (2.0 * a))
    dca_sq = (dx + dvx * t_cpa) ** 2 + (dy + dvy * t_cpa) ** 2
    closest_approach_dist = math.sqrt(max(0.0, dca_sq))

    # Case 3: Robots are diverging / moving away (b >= 0 means dp . dv >= 0)
    if b >= 0:
        return float("inf"), closest_approach_dist, closing_speed

    # Case 4: Discriminant test for future collision
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        # Trajectories pass each other without entering collision radius
        return float("inf"), closest_approach_dist, closing_speed

    # Collision roots: t1 = (-b - sqrt(D)) / (2a), t2 = (-b + sqrt(D)) / (2a)
    sqrt_d = math.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2.0 * a)

    if t1 > 0:
        return t1, closest_approach_dist, closing_speed
    else:
        # t1 <= 0 means collision occurred in the past or agent is exiting zone
        t2 = (-b + sqrt_d) / (2.0 * a)
        if t2 > 0:
            return 0.0, closest_approach_dist, closing_speed
        return float("inf"), closest_approach_dist, closing_speed


def classify_risk_tier(ttc: float) -> Tuple[int, str]:
    """
    Maps analytical Time-To-Collision to risk classification tiers:
      - 0: Safe (TTC > 3.0s or inf)
      - 1: Warning (1.0s < TTC <= 3.0s)
      - 2: High Risk (TTC <= 1.0s or actual collision)
    """
    if ttc <= TTC_WARNING_THRESHOLD:
        return RISK_LABEL_HIGH_RISK, RISK_NAMES[RISK_LABEL_HIGH_RISK]
    elif ttc <= TTC_SAFE_THRESHOLD:
        return RISK_LABEL_WARNING, RISK_NAMES[RISK_LABEL_WARNING]
    else:
        return RISK_LABEL_SAFE, RISK_NAMES[RISK_LABEL_SAFE]


class DataCollector:
    """
    Real-time data extraction engine that computes pairwise kinematic features
    and logs them to CSV with buffering for training Phase 2 & 3 ML models.
    """

    CSV_HEADERS = [
        "frame_id",
        "timestamp_sec",
        "scenario",
        "robot_i_id",
        "robot_j_id",
        "x_i",
        "y_i",
        "vx_i",
        "vy_i",
        "speed_i",
        "heading_i_rad",
        "heading_i_deg",
        "x_j",
        "y_j",
        "vx_j",
        "vy_j",
        "speed_j",
        "heading_j_rad",
        "heading_j_deg",
        "rel_pos_x",
        "rel_pos_y",
        "euclidean_distance",
        "rel_vel_x",
        "rel_vel_y",
        "rel_speed",
        "closing_speed",
        "closest_approach_dist",
        "ttc_sec",
        "risk_label",
        "risk_category",
    ]

    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        filename: str = DEFAULT_DATASET_FILENAME,
        auto_record: bool = True,
    ):
        self.dataset_dir = dataset_dir
        self.filename = filename
        self.filepath = os.path.join(self.dataset_dir, self.filename)
        self.is_recording = auto_record

        # Statistics Tracking
        self.total_records_written: int = 0
        self.label_counts: Dict[int, int] = {
            RISK_LABEL_SAFE: 0,
            RISK_LABEL_WARNING: 0,
            RISK_LABEL_HIGH_RISK: 0,
        }

        # In-memory buffer for high performance batch writes
        self._buffer: List[List[Any]] = []
        self._csv_file = None
        self._csv_writer = None

        self._initialize_file()

    def _initialize_file(self) -> None:
        """Create directory and CSV header if file doesn't already exist."""
        os.makedirs(self.dataset_dir, exist_ok=True)
        file_exists = os.path.isfile(self.filepath) and os.path.getsize(self.filepath) > 0

        self._csv_file = open(self.filepath, mode="a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)

        if not file_exists:
            self._csv_writer.writerow(self.CSV_HEADERS)
            self._csv_file.flush()

    def toggle_recording(self) -> bool:
        """Toggle recording state ON/OFF."""
        self.is_recording = not self.is_recording
        if not self.is_recording:
            self.flush()
        return self.is_recording

    def process_and_log_frame(
        self,
        frame_id: int,
        timestamp: float,
        scenario_name: str,
        robots: List[Robot],
    ) -> List[Dict[str, Any]]:
        """
        Calculates pairwise state features and TTC between every robot combination.
        Updates each robot's internal risk state and buffers records for CSV logging.

        Returns:
            List of pairwise interaction summaries for UI telemetry and visualization.
        """
        n = len(robots)
        pairwise_summaries: List[Dict[str, Any]] = []

        # Reset each robot's minimum TTC tracking for this frame
        for r in robots:
            r.min_ttc = float("inf")
            r.risk_label = RISK_LABEL_SAFE
            r.nearest_threat_id = None
            r.nearest_distance = float("inf")

        # Evaluate all pairwise combinations (i < j)
        for i in range(n):
            for j in range(i + 1, n):
                r_i = robots[i]
                r_j = robots[j]

                # Analytical TTC calculation
                ttc, dca, closing_spd = compute_analytical_ttc(
                    x1=r_i.x,
                    y1=r_i.y,
                    vx1=r_i.vx,
                    vy1=r_i.vy,
                    r1=r_i.radius,
                    x2=r_j.x,
                    y2=r_j.y,
                    vx2=r_j.vx,
                    vy2=r_j.vy,
                    r2=r_j.radius,
                )

                risk_lbl, risk_cat = classify_risk_tier(ttc)

                # Relative spatial properties
                dx = r_j.x - r_i.x
                dy = r_j.y - r_i.y
                dist = math.hypot(dx, dy)
                dvx = r_j.vx - r_i.vx
                dvy = r_j.vy - r_i.vy
                rel_spd = math.hypot(dvx, dvy)

                # Update Robot i nearest stats
                if ttc < r_i.min_ttc:
                    r_i.min_ttc = ttc
                    r_i.risk_label = risk_lbl
                    r_i.nearest_threat_id = r_j.id
                if dist < r_i.nearest_distance:
                    r_i.nearest_distance = dist

                # Update Robot j nearest stats
                if ttc < r_j.min_ttc:
                    r_j.min_ttc = ttc
                    r_j.risk_label = risk_lbl
                    r_j.nearest_threat_id = r_i.id
                if dist < r_j.nearest_distance:
                    r_j.nearest_distance = dist

                # Format TTC for storage (clip infinite values to TTC_MAX_CLIP for numeric sanity)
                logged_ttc = min(ttc, TTC_MAX_CLIP) if math.isfinite(ttc) else TTC_MAX_CLIP

                interaction_record = {
                    "pair": (r_i.id, r_j.id),
                    "dist": dist,
                    "rel_speed": rel_spd,
                    "closing_speed": closing_spd,
                    "ttc": ttc,
                    "risk_label": risk_lbl,
                    "risk_category": risk_cat,
                }
                pairwise_summaries.append(interaction_record)

                if self.is_recording:
                    row = [
                        frame_id,
                        round(timestamp, 4),
                        scenario_name,
                        r_i.id,
                        r_j.id,
                        round(r_i.x, 2),
                        round(r_i.y, 2),
                        round(r_i.vx, 2),
                        round(r_i.vy, 2),
                        round(r_i.speed, 2),
                        round(r_i.theta, 4),
                        round(r_i.heading_deg, 2),
                        round(r_j.x, 2),
                        round(r_j.y, 2),
                        round(r_j.vx, 2),
                        round(r_j.vy, 2),
                        round(r_j.speed, 2),
                        round(r_j.theta, 4),
                        round(r_j.heading_deg, 2),
                        round(dx, 2),
                        round(dy, 2),
                        round(dist, 2),
                        round(dvx, 2),
                        round(dvy, 2),
                        round(rel_spd, 2),
                        round(closing_spd, 2),
                        round(dca, 2),
                        round(logged_ttc, 3),
                        risk_lbl,
                        risk_cat,
                    ]
                    self._buffer.append(row)
                    self.total_records_written += 1
                    self.label_counts[risk_lbl] += 1

        # Periodic flush to disk
        if len(self._buffer) >= AUTO_FLUSH_INTERVAL:
            self.flush()

        return pairwise_summaries

    def flush(self) -> None:
        """Flush in-memory buffer to CSV disk file."""
        if self._buffer and self._csv_writer and self._csv_file:
            self._csv_writer.writerows(self._buffer)
            self._csv_file.flush()
            self._buffer.clear()

    def close(self) -> None:
        """Safely flush remaining records and close the CSV handle."""
        self.flush()
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    def get_stats(self) -> Dict[str, Any]:
        """Return real-time logging statistics."""
        total = self.total_records_written
        safe_cnt = self.label_counts[RISK_LABEL_SAFE]
        warn_cnt = self.label_counts[RISK_LABEL_WARNING]
        crit_cnt = self.label_counts[RISK_LABEL_HIGH_RISK]

        return {
            "total_records": total,
            "safe_count": safe_cnt,
            "warning_count": warn_cnt,
            "high_risk_count": crit_cnt,
            "safe_pct": (safe_cnt / total * 100.0) if total > 0 else 0.0,
            "warning_pct": (warn_cnt / total * 100.0) if total > 0 else 0.0,
            "high_risk_pct": (crit_cnt / total * 100.0) if total > 0 else 0.0,
            "is_recording": self.is_recording,
            "filepath": self.filepath,
        }

    def load_dataframe(self) -> pd.DataFrame:
        """Load the logged dataset directly into a Pandas DataFrame."""
        self.flush()
        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            return pd.read_csv(self.filepath)
        return pd.DataFrame(columns=self.CSV_HEADERS)

"""
GNN Online Inference Controller for Multi-Robot Navigation (Phase 4 Stage B).

Responsibilities:
- Loads Phase 3 model checkpoint and normalization scalers exactly once during initialization.
- Constructs live multi-agent scene graphs directly from autonomous robot states in memory.
- Executes batched GNN edge classification with configurable inference frequency (e.g., 60 Hz, 30 Hz).
- Returns structured pairwise risk predictions and per-robot threat summaries.
- Provides low-latency caching between inference steps.
"""

import os
import sys
import time
import math
import pickle
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Ensure src and root are in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    RISK_LABEL_SAFE,
    RISK_LABEL_WARNING,
    RISK_LABEL_HIGH_RISK,
    RISK_NAMES,
)
from robot_state import Robot
from src.model_gnn import CollisionRiskGNN


@dataclass
class EdgeRiskPrediction:
    """Pairwise collision-risk prediction for a directed edge (u -> v)."""
    src_robot_id: int
    dst_robot_id: int
    risk_label: int                      # 0: Safe, 1: Warning, 2: High Risk
    risk_category: str                   # 'SAFE', 'WARNING', 'HIGH_RISK'
    probabilities: Tuple[float, float, float]  # (P_safe, P_warn, P_high)
    confidence: float                    # Max probability
    distance: float                      # Euclidean distance between centers
    rel_speed: float                     # Relative speed ||dv||
    closing_speed: float                 # Rate of approach
    closest_approach_dist: float         # Projected DCA


@dataclass
class RobotRiskSummary:
    """Aggregated topological risk state for an individual autonomous robot."""
    robot_id: int
    max_risk_label: int                  # Highest risk tier involving this robot
    max_risk_category: str               # 'SAFE', 'WARNING', 'HIGH_RISK'
    highest_threat_id: Optional[int]     # ID of the robot posing the highest risk
    highest_threat_prob: float           # Probability of the highest risk tier
    highest_threat_distance: float       # Distance to the primary threat


@dataclass
class InferenceFrameResult:
    """Complete prediction output for a simulation time step."""
    frame_id: int
    sim_time: float
    edge_predictions: List[EdgeRiskPrediction]
    robot_summaries: Dict[int, RobotRiskSummary]
    is_fresh_inference: bool             # True if model evaluated; False if served from cache
    inference_latency_ms: float          # Time spent in GNN forward pass (ms)


class GNNInferenceController:
    """
    High-performance online inference controller for Phase 3 CollisionRiskGNN.
    Loads neural network weights and normalization scalers once, constructs
    scene graphs from live simulation objects, and supports configurable inference frequency.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/gnn_interaction_model.pth",
        scalers_path: str = "checkpoints/gnn_scalers.pkl",
        inference_interval: int = 1,
        interaction_radius: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initializes the controller and loads weights/scalers strictly once.

        Args:
            checkpoint_path: Path to trained GNN checkpoint (.pth).
            scalers_path: Path to pickled node/edge StandardScalers (.pkl).
            inference_interval: Execute GNN every K frames (1 = every frame; 2 = every 2nd frame).
            interaction_radius: Optional distance cutoff (in px) for creating interaction edges.
            device: Compute device (CPU, MPS, CUDA). If None, auto-selects optimal device.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GNN checkpoint not found: {checkpoint_path}")
        if not os.path.exists(scalers_path):
            raise FileNotFoundError(f"GNN scalers not found: {scalers_path}")

        self.checkpoint_path = checkpoint_path
        self.scalers_path = scalers_path
        self.inference_interval = max(1, int(inference_interval))
        self.interaction_radius = interaction_radius

        # 1. Resolve Device
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 2. Load Scalers (strictly once)
        with open(self.scalers_path, "rb") as f:
            scalers = pickle.load(f)
        self.node_scaler = scalers["node_scaler"]
        self.edge_scaler = scalers["edge_scaler"]

        # 3. Load Checkpoint and Instantiate Model (strictly once)
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.node_dim = ckpt.get("node_dim", 6)
        self.edge_dim = ckpt.get("edge_dim", 8)
        self.hidden_dim = ckpt.get("hidden_dim", 64)
        self.checkpoint_epoch = ckpt.get("epoch", -1)

        self.model = CollisionRiskGNN(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            num_classes=3,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 4. State & Performance Metrics
        self.last_result: Optional[InferenceFrameResult] = None
        self.total_calls: int = 0
        self.total_model_inferences: int = 0
        self.total_cache_hits: int = 0
        self.total_latency_ms: float = 0.0

    def set_inference_interval(self, interval: int) -> None:
        """Dynamically configure the frame evaluation cadence."""
        self.inference_interval = max(1, int(interval))

    def predict_frame(
        self,
        robots: List[Robot],
        frame_id: int = 0,
        sim_time: float = 0.0,
        force_inference: bool = False,
    ) -> InferenceFrameResult:
        """
        Generates pairwise risk predictions and per-robot risk summaries.
        If `frame_id % inference_interval != 0` and not `force_inference`,
        returns cached predictions from the previous forward pass.

        Args:
            robots: List of active Robot objects in simulation.
            frame_id: Current simulation frame index.
            sim_time: Current simulation elapsed time in seconds.
            force_inference: If True, bypasses frequency throttling.

        Returns:
            InferenceFrameResult containing all edge predictions and robot summaries.
        """
        self.total_calls += 1
        num_robots = len(robots)

        # Degenerate case: < 2 robots implies no interactions
        if num_robots < 2:
            empty_result = InferenceFrameResult(
                frame_id=frame_id,
                sim_time=sim_time,
                edge_predictions=[],
                robot_summaries={
                    r.id: RobotRiskSummary(
                        robot_id=r.id,
                        max_risk_label=RISK_LABEL_SAFE,
                        max_risk_category=RISK_NAMES[RISK_LABEL_SAFE],
                        highest_threat_id=None,
                        highest_threat_prob=1.0,
                        highest_threat_distance=float("inf"),
                    )
                    for r in robots
                },
                is_fresh_inference=False,
                inference_latency_ms=0.0,
            )
            self.last_result = empty_result
            return empty_result

        # Determine whether to execute model or return cache
        should_infer = (
            force_inference
            or (self.last_result is None)
            or (frame_id % self.inference_interval == 0)
        )

        if not should_infer and self.last_result is not None:
            self.total_cache_hits += 1
            # Return cached result with updated frame_id and timestamp
            cached = InferenceFrameResult(
                frame_id=frame_id,
                sim_time=sim_time,
                edge_predictions=self.last_result.edge_predictions,
                robot_summaries=self.last_result.robot_summaries,
                is_fresh_inference=False,
                inference_latency_ms=0.0,
            )
            return cached

        # Execute Fresh Online Graph Inference
        t_start = time.perf_counter()

        # 1. Build Node Feature Matrix (Ordered strictly by robot ID)
        sorted_robots = sorted(robots, key=lambda r: r.id)
        id_to_idx = {r.id: idx for idx, r in enumerate(sorted_robots)}

        node_features = np.zeros((num_robots, self.node_dim), dtype=np.float32)
        for idx, r in enumerate(sorted_robots):
            node_features[idx] = [r.x, r.y, r.vx, r.vy, r.theta, r.speed]

        # 2. Build Directed Edge Indices & Relational Attributes
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_attrs_raw: List[List[float]] = []
        edge_meta: List[Tuple[int, int, float, float, float, float]] = []

        for i in range(num_robots):
            for j in range(num_robots):
                if i == j:
                    continue

                r_i = sorted_robots[i]
                r_j = sorted_robots[j]

                dx = r_j.x - r_i.x
                dy = r_j.y - r_i.y
                dist = math.hypot(dx, dy)

                if self.interaction_radius is not None and dist > self.interaction_radius:
                    continue

                dvx = r_j.vx - r_i.vx
                dvy = r_j.vy - r_i.vy
                rel_spd = math.hypot(dvx, dvy)

                closing_spd = -(dx * dvx + dy * dvy) / dist if dist > 1e-6 else 0.0

                # Distance of closest approach (DCA)
                a = dvx * dvx + dvy * dvy
                b = 2.0 * (dx * dvx + dy * dvy)
                t_cpa = max(0.0, -b / (2.0 * a)) if a >= 1e-8 else 0.0
                dca_sq = (dx + dvx * t_cpa) ** 2 + (dy + dvy * t_cpa) ** 2
                dca = math.sqrt(max(0.0, dca_sq))

                edge_src.append(i)
                edge_dst.append(j)
                edge_attrs_raw.append([dx, dy, dist, dvx, dvy, rel_spd, closing_spd, dca])
                edge_meta.append((r_i.id, r_j.id, dist, rel_spd, closing_spd, dca))

        num_edges = len(edge_src)
        if num_edges == 0:
            # No interacting pairs within radius
            empty_result = InferenceFrameResult(
                frame_id=frame_id,
                sim_time=sim_time,
                edge_predictions=[],
                robot_summaries={
                    r.id: RobotRiskSummary(
                        robot_id=r.id,
                        max_risk_label=RISK_LABEL_SAFE,
                        max_risk_category=RISK_NAMES[RISK_LABEL_SAFE],
                        highest_threat_id=None,
                        highest_threat_prob=1.0,
                        highest_threat_distance=float("inf"),
                    )
                    for r in sorted_robots
                },
                is_fresh_inference=True,
                inference_latency_ms=(time.perf_counter() - t_start) * 1000.0,
            )
            self.last_result = empty_result
            return empty_result

        # 3. Standardize Features using Training Scalers
        node_features_norm = self.node_scaler.transform(node_features)
        edge_attrs_norm = self.edge_scaler.transform(np.array(edge_attrs_raw, dtype=np.float32))

        # 4. Convert to PyTorch Tensors
        x_tensor = torch.tensor(node_features_norm, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=self.device)
        edge_attr_tensor = torch.tensor(edge_attrs_norm, dtype=torch.float32, device=self.device)

        # 5. GNN Forward Inference
        with torch.no_grad():
            logits = self.model(x_tensor, edge_index, edge_attr_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.total_model_inferences += 1
        self.total_latency_ms += latency_ms

        # 6. Parse Edge Predictions & Robot Summaries
        edge_predictions: List[EdgeRiskPrediction] = []
        # Pre-initialize per-robot tracking
        robot_risk_track: Dict[int, Dict[str, Any]] = {
            r.id: {
                "max_risk": RISK_LABEL_SAFE,
                "threat_id": None,
                "threat_prob": 0.0,
                "threat_dist": float("inf"),
            }
            for r in sorted_robots
        }

        for k in range(num_edges):
            src_id, dst_id, dist, rel_spd, closing_spd, dca = edge_meta[k]
            risk_lbl = int(preds[k])
            prob_tuple = (float(probs[k, 0]), float(probs[k, 1]), float(probs[k, 2]))
            conf = float(probs[k, risk_lbl])

            pred = EdgeRiskPrediction(
                src_robot_id=src_id,
                dst_robot_id=dst_id,
                risk_label=risk_lbl,
                risk_category=RISK_NAMES[risk_lbl],
                probabilities=prob_tuple,
                confidence=conf,
                distance=dist,
                rel_speed=rel_spd,
                closing_speed=closing_spd,
                closest_approach_dist=dca,
            )
            edge_predictions.append(pred)

            # Update Source Robot threat assessment
            track = robot_risk_track[src_id]
            if risk_lbl > track["max_risk"]:
                track["max_risk"] = risk_lbl
                track["threat_id"] = dst_id
                track["threat_prob"] = prob_tuple[risk_lbl]
                track["threat_dist"] = dist
            elif risk_lbl == track["max_risk"] and dist < track["threat_dist"]:
                # If equal risk tier, prioritize the closer physical agent
                track["threat_id"] = dst_id
                track["threat_prob"] = prob_tuple[risk_lbl]
                track["threat_dist"] = dist

        robot_summaries: Dict[int, RobotRiskSummary] = {}
        for r_id, tr in robot_risk_track.items():
            robot_summaries[r_id] = RobotRiskSummary(
                robot_id=r_id,
                max_risk_label=tr["max_risk"],
                max_risk_category=RISK_NAMES[tr["max_risk"]],
                highest_threat_id=tr["threat_id"],
                highest_threat_prob=tr["threat_prob"],
                highest_threat_distance=tr["threat_dist"],
            )

        frame_result = InferenceFrameResult(
            frame_id=frame_id,
            sim_time=sim_time,
            edge_predictions=edge_predictions,
            robot_summaries=robot_summaries,
            is_fresh_inference=True,
            inference_latency_ms=latency_ms,
        )
        self.last_result = frame_result
        return frame_result

    def update_robot_risk_states(
        self,
        robots: List[Robot],
        frame_result: InferenceFrameResult,
    ) -> None:
        """
        Directly syncs robot kinematic telemetry fields (`risk_label`,
        `nearest_threat_id`, `nearest_distance`) with the GNN prediction.
        """
        for r in robots:
            if r.id in frame_result.robot_summaries:
                summary = frame_result.robot_summaries[r.id]
                r.risk_label = summary.max_risk_label
                r.nearest_threat_id = summary.highest_threat_id
                r.nearest_distance = summary.highest_threat_distance

    def get_metrics(self) -> Dict[str, Any]:
        """Returns execution and performance metrics."""
        avg_latency = (
            self.total_latency_ms / max(1, self.total_model_inferences)
        )
        cache_rate = (
            (self.total_cache_hits / max(1, self.total_calls)) * 100.0
        )
        return {
            "total_calls": self.total_calls,
            "total_model_inferences": self.total_model_inferences,
            "total_cache_hits": self.total_cache_hits,
            "cache_hit_rate_pct": cache_rate,
            "avg_latency_ms": avg_latency,
            "inference_interval": self.inference_interval,
            "device": str(self.device),
        }

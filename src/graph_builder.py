"""
Graph Data Engineering Pipeline for Multi-Robot Interaction Modeling (Phase 3).
Converts chronological simulation trajectories into dynamic PyTorch Geometric (PyG) graphs.

Graph Topology:
- Nodes: Individual autonomous robots. Node features encompass absolute kinematics:
  [x, y, vx, vy, theta, speed].
- Edges: Dynamic interaction edges between robot pairs. Edge attributes encompass
  relative spatial and kinematic features:
  [rel_pos_x, rel_pos_y, euclidean_distance, rel_vel_x, rel_vel_y, rel_speed, closing_speed, closest_approach_dist].
- Targets: Discrete collision risk tiers for each edge:
  0: Safe (TTC > 3s), 1: Warning (1s < TTC <= 3s), 2: High Risk (TTC <= 1s).
"""

import os
import sys
import pickle
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

NODE_FEATURE_NAMES = ["x", "y", "vx", "vy", "theta_rad", "speed"]
EDGE_FEATURE_NAMES = [
    "rel_pos_x",
    "rel_pos_y",
    "euclidean_distance",
    "rel_vel_x",
    "rel_vel_y",
    "rel_speed",
    "closing_speed",
    "closest_approach_dist",
]


@dataclass
class GraphDatasetBundle:
    """Container holding PyG DataLoaders, scalers, class weights, and graph lists."""
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    node_scaler: StandardScaler
    edge_scaler: StandardScaler
    class_weights: torch.Tensor
    train_graphs: List[Data]
    val_graphs: List[Data]
    test_graphs: List[Data]
    node_dim: int
    edge_dim: int
    num_classes: int = 3


def build_graphs_from_dataframe(
    df: pd.DataFrame,
    interaction_radius: Optional[float] = None,
) -> List[Data]:
    """
    Groups raw tabular simulation frames by (scenario, frame_id) and constructs
    PyTorch Geometric Data instances with bidirectional edges and pairwise targets.

    Args:
        df: DataFrame containing pairwise simulation telemetry.
        interaction_radius: Optional distance cutoff (in pixels). If None, all interacting
                            pairs in the bounded arena form edges.

    Returns:
        List of PyTorch Geometric Data objects.
    """
    graph_list: List[Data] = []

    # Group by unique time frames
    frame_groups = df.groupby(["scenario", "frame_id"], sort=False)

    for (scenario, frame_id), group in frame_groups:
        # 1. Reconstruct unique robot node features for this frame
        # In our pairwise dataset, each row has robot_i and robot_j
        robots_dict: Dict[int, List[float]] = {}

        for _, row in group.iterrows():
            r_i = int(row["robot_i_id"])
            if r_i not in robots_dict:
                robots_dict[r_i] = [
                    row["x_i"],
                    row["y_i"],
                    row["vx_i"],
                    row["vy_i"],
                    row["heading_i_rad"],
                    row["speed_i"],
                ]

            r_j = int(row["robot_j_id"])
            if r_j not in robots_dict:
                robots_dict[r_j] = [
                    row["x_j"],
                    row["y_j"],
                    row["vx_j"],
                    row["vy_j"],
                    row["heading_j_rad"],
                    row["speed_j"],
                ]

        # Order nodes strictly by robot ID: 0, 1, 2, ...
        sorted_ids = sorted(robots_dict.keys())
        id_to_idx = {rid: idx for idx, rid in enumerate(sorted_ids)}
        node_features = np.array([robots_dict[rid] for rid in sorted_ids], dtype=np.float32)

        # 2. Construct Edges, Edge Attributes, and Edge Targets
        edge_indices_src = []
        edge_indices_dst = []
        edge_attrs = []
        edge_targets = []
        edge_pair_ids = []  # Original (r_i, r_j) for tracking / test alignment
        is_canonical_edge = []  # True for i < j (canonical pairs for baseline comparison)

        for _, row in group.iterrows():
            r_i = int(row["robot_i_id"])
            r_j = int(row["robot_j_id"])
            u = id_to_idx[r_i]
            v = id_to_idx[r_j]

            dist = float(row["euclidean_distance"])
            if interaction_radius is not None and dist > interaction_radius:
                continue

            target_lbl = int(row["risk_label"])

            # Forward edge (u -> v): Robot i to Robot j
            attr_forward = [
                float(row["rel_pos_x"]),
                float(row["rel_pos_y"]),
                dist,
                float(row["rel_vel_x"]),
                float(row["rel_vel_y"]),
                float(row["rel_speed"]),
                float(row["closing_speed"]),
                float(row["closest_approach_dist"]),
            ]
            edge_indices_src.append(u)
            edge_indices_dst.append(v)
            edge_attrs.append(attr_forward)
            edge_targets.append(target_lbl)
            edge_pair_ids.append((r_i, r_j))
            is_canonical_edge.append(True)

            # Reverse bidirectional edge (v -> u): Robot j to Robot i
            # Relative displacement and velocity vectors are negated; scalars stay identical
            attr_reverse = [
                -float(row["rel_pos_x"]),
                -float(row["rel_pos_y"]),
                dist,
                -float(row["rel_vel_x"]),
                -float(row["rel_vel_y"]),
                float(row["rel_speed"]),
                float(row["closing_speed"]),
                float(row["closest_approach_dist"]),
            ]
            edge_indices_src.append(v)
            edge_indices_dst.append(u)
            edge_attrs.append(attr_reverse)
            edge_targets.append(target_lbl)
            edge_pair_ids.append((r_j, r_i))
            is_canonical_edge.append(False)

        if not edge_indices_src:
            continue

        edge_index = torch.tensor([edge_indices_src, edge_indices_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        y = torch.tensor(edge_targets, dtype=torch.long)
        x = torch.tensor(node_features, dtype=torch.float32)
        canonical_mask = torch.tensor(is_canonical_edge, dtype=torch.bool)

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            canonical_mask=canonical_mask,
            scenario=scenario,
            frame_id=frame_id,
            num_nodes=len(sorted_ids),
        )
        graph_list.append(graph_data)

    return graph_list


def load_and_preprocess_graphs(
    csv_path: str = "dataset/collision_risk_dataset.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 64,
    interaction_radius: Optional[float] = None,
    scaler_save_path: Optional[str] = "checkpoints/gnn_scalers.pkl",
) -> GraphDatasetBundle:
    """
    Ingests tabular CSV, converts into scene graphs, performs chronological splitting
    by scenario, standardizes node and edge features with training statistics,
    and returns PyG DataLoaders and metadata.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Chronological splitting by scenario at the frame level
    train_dfs = []
    val_dfs = []
    test_dfs = []

    scenario_groups = df.groupby("scenario", sort=False)
    for scen_name, group in scenario_groups:
        unique_frames = group["frame_id"].unique()
        n_frames = len(unique_frames)
        n_train = int(n_frames * train_ratio)
        n_val = int(n_frames * val_ratio)

        train_frame_set = set(unique_frames[:n_train])
        val_frame_set = set(unique_frames[n_train:n_train + n_val])
        test_frame_set = set(unique_frames[n_train + n_val:])

        train_dfs.append(group[group["frame_id"].isin(train_frame_set)])
        val_dfs.append(group[group["frame_id"].isin(val_frame_set)])
        test_dfs.append(group[group["frame_id"].isin(test_frame_set)])

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # 2. Build PyG graphs for each partition
    train_graphs = build_graphs_from_dataframe(train_df, interaction_radius=interaction_radius)
    val_graphs = build_graphs_from_dataframe(val_df, interaction_radius=interaction_radius)
    test_graphs = build_graphs_from_dataframe(test_df, interaction_radius=interaction_radius)

    # 3. Fit StandardScalers strictly on the Training set
    # Collect training node and edge attributes
    all_train_x = np.concatenate([g.x.numpy() for g in train_graphs], axis=0)
    all_train_e = np.concatenate([g.edge_attr.numpy() for g in train_graphs], axis=0)

    node_scaler = StandardScaler()
    edge_scaler = StandardScaler()

    node_scaler.fit(all_train_x)
    edge_scaler.fit(all_train_e)

    # Optionally persist scalers
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        with open(scaler_save_path, "wb") as f:
            pickle.dump({"node_scaler": node_scaler, "edge_scaler": edge_scaler}, f)

    # Transform graphs in place
    for g in train_graphs:
        g.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float32)
        g.edge_attr = torch.tensor(edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float32)

    for g in val_graphs:
        g.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float32)
        g.edge_attr = torch.tensor(edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float32)

    for g in test_graphs:
        g.x = torch.tensor(node_scaler.transform(g.x.numpy()), dtype=torch.float32)
        g.edge_attr = torch.tensor(edge_scaler.transform(g.edge_attr.numpy()), dtype=torch.float32)

    # 4. Compute Class Weights on canonical training edges
    train_labels = np.concatenate([g.y[g.canonical_mask].numpy() for g in train_graphs], axis=0)
    classes, counts = np.unique(train_labels, return_counts=True)
    total_samples = len(train_labels)
    num_classes = len(classes)

    weights = np.zeros(num_classes, dtype=np.float32)
    for c, count in zip(classes, counts):
        weights[c] = total_samples / (num_classes * count)

    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # 5. Build PyTorch Geometric DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    return GraphDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        node_scaler=node_scaler,
        edge_scaler=edge_scaler,
        class_weights=class_weights_tensor,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        node_dim=len(NODE_FEATURE_NAMES),
        edge_dim=len(EDGE_FEATURE_NAMES),
        num_classes=num_classes,
    )


if __name__ == "__main__":
    bundle = load_and_preprocess_graphs()
    print("Graph Dataset Bundle Created Successfully:")
    print(f" Train Graphs: {len(bundle.train_graphs):,} | Val Graphs: {len(bundle.val_graphs):,} | Test Graphs: {len(bundle.test_graphs):,}")
    print(f" Node Feature Dim: {bundle.node_dim} | Edge Feature Dim: {bundle.edge_dim}")
    print(f" Class Weights: Safe={bundle.class_weights[0]:.2f}, Warn={bundle.class_weights[1]:.2f}, HighRisk={bundle.class_weights[2]:.2f}")

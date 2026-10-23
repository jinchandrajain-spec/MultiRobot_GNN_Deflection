"""
Attentive Graph Neural Network (GNN) for Multi-Agent Collision Risk Modeling.
Phase 3: GNN Interaction Modeling.

Architecture:
- Node Feature Embedding with Layer Normalization and GELU activation.
- Edge Attribute Feature Projection with Layer Normalization and GELU activation.
- Multi-Head Graph Attention Network (GATv2Conv) conditioning attention weights on
  relative displacement, approach velocity, and dynamic distance.
- LayerNorm and residual connections ensuring stable gradient flow without mini-batch drift.
- Topological Edge Classifier Head that aggregates source, target, contextual difference,
  relational dot product, and exact raw edge attributes.
- Output logits of shape (num_edges, 3).
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class CollisionRiskGNN(nn.Module):
    """
    Attentive Graph Neural Network predicting pairwise collision risk tiers
    by modeling collective multi-agent topological context.
    """

    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 8,
        hidden_dim: int = 64,
        edge_hidden_dim: int = 32,
        num_classes: int = 3,
        num_layers: int = 2,
        heads: int = 4,
        dropout_p: float = 0.20,
    ):
        super(CollisionRiskGNN, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_p = dropout_p

        # 1. Node & Edge Feature Encoders (LayerNorm avoids mini-batch running mean drift)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
        )

        # 2. Attentive GATv2 Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=heads,
                edge_dim=edge_hidden_dim,
                concat=True,
                dropout=dropout_p,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(p=dropout_p))

        # 3. Topological Edge Classifier Head
        # Concatenates: [h_u, h_v, |h_u - h_v|, h_u * h_v, e_proj, edge_attr_raw]
        edge_classifier_in_dim = hidden_dim * 4 + edge_hidden_dim + edge_dim

        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_classifier_in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for linear projections."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward propagation through GAT interaction layers and edge classification head.

        Args:
            x: Node feature tensor of shape (num_nodes, node_dim).
            edge_index: Graph connectivity matrix of shape (2, num_edges).
            edge_attr: Edge attributes tensor of shape (num_edges, edge_dim).

        Returns:
            Logits tensor of shape (num_edges, num_classes).
        """
        # 1. Project initial node and edge features
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)

        # 2. Attentive message passing with residual connections
        for i in range(self.num_layers):
            h_res = h
            h_conv = self.convs[i](h, edge_index, edge_attr=e)
            h_norm = self.norms[i](h_conv)
            h_act = F.gelu(h_norm)
            h = h_res + self.dropouts[i](h_act)

        # 3. Construct topological edge representations
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        h_src = h[src_nodes]
        h_dst = h[dst_nodes]
        h_diff = torch.abs(h_src - h_dst)
        h_prod = h_src * h_dst

        # Combined edge representation incorporates relational edge features,
        # global neighborhood contextual state, and exact raw kinematic attributes
        z_edge = torch.cat([h_src, h_dst, h_diff, h_prod, e, edge_attr], dim=-1)

        # 4. Project to risk class logits
        logits = self.edge_classifier(z_edge)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Computes edge class probabilities via Softmax."""
        self.eval()
        logits = self.forward(x, edge_index, edge_attr)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts discrete edge risk labels via Argmax."""
        self.eval()
        logits = self.forward(x, edge_index, edge_attr)
        return torch.argmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        param_cnt = self.count_parameters()
        return (
            f"CollisionRiskGNN(\n"
            f"  node_dim={self.node_dim}, edge_dim={self.edge_dim}, hidden_dim={self.hidden_dim},\n"
            f"  num_layers={self.num_layers}, heads={self.heads}, dropout_p={self.dropout_p},\n"
            f"  total_trainable_params={param_cnt:,}\n"
            f")"
        )

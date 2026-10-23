"""
Baseline PyTorch Multi-Layer Perceptron (MLP) for Collision Risk Prediction.
Phase 2: Baseline Predictive Engine.

Architecture:
- Modular 4-layer Feedforward Neural Network.
- Batch Normalization (BatchNorm1d) on hidden layers for covariate shift stabilization.
- Dropout regularization (p=0.25) to prevent overfitting on minority classes.
- GELU (Gaussian Error Linear Unit) non-linear activation functions.
- Output logits over 3 risk classes (0: Safe, 1: Warning, 2: High Risk).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CollisionRiskMLP(nn.Module):
    """
    Multi-Layer Perceptron classifier predicting multi-agent collision risk tiers
    from normalized relative kinematic features.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 3,
        dropout_p: float = 0.25,
        activation: str = "gelu",
    ):
        super(CollisionRiskMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_p = dropout_p

        # Select activation function
        if activation.lower() == "gelu":
            act_fn = nn.GELU
        elif activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "silu" or activation.lower() == "swish":
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Construct Feedforward Network Layers
        layers: List[nn.Module] = []
        in_features = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(p=dropout_p))
            in_features = hidden_dim

        # Final classification linear projection to logits
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier_head = nn.Linear(in_features, num_classes)

        # Initialize weights with Kaiming Normal initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming normal initialization to linear layers and zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing unnormalized class logits.
        Args:
            x: Input feature tensor of shape (batch_size, input_dim)
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        logits = self.classifier_head(features)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes class probabilities via Softmax."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts discrete class labels via Argmax."""
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        param_cnt = self.count_parameters()
        return (
            f"CollisionRiskMLP(\n"
            f"  input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, num_classes={self.num_classes},\n"
            f"  dropout_p={self.dropout_p}, total_trainable_params={param_cnt:,}\n"
            f")"
        )

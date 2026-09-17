"""PyTorch Neural Network Architecture for Intersection Vehicle Trajectory Prediction.

Predicts a 1.50s future trajectory horizon (6 waypoints at 0.25s intervals)
from a 5-step (0.5s) history of 10 Hz V2V telemetry.
"""

import torch
import torch.nn as nn


class IntersectionTrajectoryMLP(nn.Module):
    """Multi-Layer Perceptron for urban vehicle trajectory forecasting."""

    def __init__(self, input_dim: int = 40, hidden_dim: int = 64, output_dim: int = 12):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, 40) or (40,)
            Flattened history features.

        Returns
        -------
        torch.Tensor of shape (batch_size, 12)
            Predicted [dx_0.25, dy_0.25, dx_0.50, dy_0.50, ..., dx_1.50, dy_1.50].
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.net(x).squeeze(0)
        return self.net(x)

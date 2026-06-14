from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 60, action_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

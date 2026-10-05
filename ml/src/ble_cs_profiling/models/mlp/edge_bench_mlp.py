"""Reines Fully-Connected-MLP als Speicher-/Cache-Benchmark für die MCU.

Ohne Convs: misst vor allem, wie schnell Gewichte aus Flash/SRAM gelesen werden.
"""

import torch
from torch import nn


class EdgeBenchMLP(nn.Module):
    """300 -> 64 -> 32 -> 1, flacher IQ-Input wie bei allen Pipeline-Modellen."""

    def __init__(self, in_features: int = 300, num_outputs: int = 1) -> None:
        super().__init__()
        # 300 Features -> 64 Hidden Nodes (19.200 Gewichts-Bytes in INT8)
        self.fc1 = nn.Linear(in_features, 64, bias=True)
        self.relu1 = nn.ReLU()
        # 64 -> 32 Hidden Nodes (2.048 Gewichts-Bytes)
        self.fc2 = nn.Linear(64, 32, bias=True)
        self.relu2 = nn.ReLU()
        # 32 -> Regressionsausgang (32 Gewichts-Bytes)
        self.fc3 = nn.Linear(32, num_outputs, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

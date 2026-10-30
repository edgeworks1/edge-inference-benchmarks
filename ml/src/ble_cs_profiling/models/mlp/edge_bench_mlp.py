"""Pure fully connected MLP: the smallest pipeline model and a memory benchmark for the MCU.

Without convolutions the runtime is dominated by reading weights from flash, which makes it
a good probe for memory bandwidth and cache effects.
"""

import torch
from torch import nn


class EdgeBenchMLP(nn.Module):
    """300 -> 64 -> 32 -> 1 with ReLU, on the flat IQ input like all pipeline models."""

    def __init__(self, in_features: int = 300, num_outputs: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64, bias=True)  # 19,200 weight bytes in INT8
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32, bias=True)  # 2,048 weight bytes
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_outputs, bias=True)  # 32 weight bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the distance for a batch of flat IQ samples ``(batch, 300)``."""
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

"""Shared building blocks and network skeleton of the MobileNetV3 variants.

One BLE sample (300 values) is treated as a single-channel image of 4 x 75 pixels: the rows
are local I, local Q, remote I and remote Q, the columns the 75 IQ sample positions.

The variants (small, micro, bad SIMD) only differ in their block table and a few sizes;
everything else lives here. Attribute names (``features``, ``avgpool``, ``head``, ``block``)
define the keys of the saved weights and must not change, or old checkpoints stop loading.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

BlockActivation = Literal["hardswish", "relu"]

# One row of a block table: (kernel, expanded_channels, out_channels, use_se, use_hs, stride)
BlockRow = tuple[int, int, int, bool, bool, int]


def make_divisible(value: float, divisor: int = 8) -> int:
    """Round a channel count to the nearest multiple of ``divisor``, losing at most 10 %.

    Multiples of 8 keep SIMD lanes and NPU tiles fully used (from the original MobileNet code).
    """
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


def default_squeeze_channels(expanded_channels: int) -> int:
    """Bottleneck width of a squeeze-and-excitation block: a quarter, rounded to a multiple of 8."""
    return make_divisible(expanded_channels / 4)


@dataclass(frozen=True)
class InvertedResidualConfig:
    """Hyperparameters of one inverted residual block (one row of a block table)."""

    kernel: int
    expanded_channels: int
    out_channels: int
    use_se: bool  # add a squeeze-and-excitation block
    use_hs: bool  # HardSwish instead of ReLU
    stride: int


def block_configs(
    rows: Sequence[BlockRow], scale: Callable[[int], int] = lambda c: c
) -> list[InvertedResidualConfig]:
    """Turn a block table into configs, scaling both channel counts with ``scale``."""
    return [
        InvertedResidualConfig(kernel, scale(expanded), scale(out), use_se, use_hs, stride)
        for kernel, expanded, out, use_se, use_hs, stride in rows
    ]


class ConvNormActivation(nn.Sequential):
    """Convolution without bias, batch norm, activation; padding keeps the size at stride 1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,  # the following batch norm has its own bias
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )


class SqueezeExcitation2D(nn.Module):
    """Channel attention: weights every channel by a factor learned from its global average."""

    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, kernel_size=1)
        self.scale_activation = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each channel of ``x`` by its attention weight in [0, 1]."""
        scale = self.fc2(self.relu(self.fc1(self.avgpool(x))))
        return x * self.scale_activation(scale)


class InvertedResidual2D(nn.Module):
    """MobileNetV3 block: expand (1x1) -> depthwise (k x k) -> optional SE -> project (1x1).

    Adds the input to the output (residual) when shape and channel count stay the same.
    """

    def __init__(
        self,
        in_channels: int,
        config: InvertedResidualConfig,
        activation: type[nn.Module],
        squeeze_channels: int,
    ) -> None:
        super().__init__()
        if config.stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        expanded = config.expanded_channels

        layers: list[nn.Module] = []
        if expanded != in_channels:
            layers.append(ConvNormActivation(in_channels, expanded, 1, activation=activation))
        layers.append(
            ConvNormActivation(
                expanded,
                expanded,
                config.kernel,
                stride=config.stride,
                groups=expanded,
                activation=activation,
            )
        )
        if config.use_se:
            layers.append(SqueezeExcitation2D(expanded, squeeze_channels))
        # Linear projection: no activation, it would destroy information in the narrow output
        layers.append(
            nn.Sequential(
                nn.Conv2d(expanded, config.out_channels, 1, bias=False),
                nn.BatchNorm2d(config.out_channels),
            )
        )

        self.block = nn.Sequential(*layers)
        self.use_residual = config.stride == 1 and in_channels == config.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block, plus the residual connection if shapes allow it."""
        result = self.block(x)
        return result + x if self.use_residual else result


class MobileNetV3Regressor2D(nn.Module):
    """MobileNetV3 with a regression head: stem, blocks, 1x1 conv, pooling, MLP head."""

    def __init__(
        self,
        blocks: Sequence[InvertedResidualConfig],
        stem_channels: int,
        stem_stride: int,
        last_conv_channels: int,
        hidden_channels: int,
        dropout: float,
        image_height: int = 4,
        image_width: int = 75,
        num_outputs: int = 1,
        block_activation: BlockActivation = "hardswish",
        squeeze_channels: Callable[[int], int] = default_squeeze_channels,
    ) -> None:
        super().__init__()
        if block_activation not in ("hardswish", "relu"):
            raise ValueError("block_activation must be 'hardswish' or 'relu'")
        self.image_height = image_height
        self.image_width = image_width

        layers: list[nn.Module] = [
            ConvNormActivation(1, stem_channels, 3, stride=stem_stride, activation=nn.Hardswish)
        ]
        channels = stem_channels
        for config in blocks:
            activation = _block_activation(config, block_activation)
            layers.append(
                InvertedResidual2D(
                    channels, config, activation, squeeze_channels(config.expanded_channels)
                )
            )
            channels = config.out_channels
        layers.append(ConvNormActivation(channels, last_conv_channels, 1, activation=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(last_conv_channels, hidden_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, num_outputs),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the distance for a batch of flat (300) or image-shaped (4 x 75) samples."""
        x = self.features(self._as_image(x))
        return self.head(torch.flatten(self.avgpool(x), 1))

    def _as_image(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(batch, 300)`` or ``(batch, 4, 75)`` input to ``(batch, 1, 4, 75)``."""
        expected_shape = (1, self.image_height, self.image_width)
        if x.ndim == 2:
            x = x.reshape(x.shape[0], *expected_shape)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected input (batch, 300), (batch, 4, 75) or (batch, 1, 4, 75), "
                f"got {tuple(x.shape)}"
            )
        return x

    def _initialize_weights(self) -> None:
        """Standard MobileNetV3 initialization (as in torchvision)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)


def _block_activation(
    config: InvertedResidualConfig, block_activation: BlockActivation
) -> type[nn.Module]:
    """HardSwish where the block table asks for it, unless all blocks are forced to ReLU."""
    if block_activation == "relu":
        return nn.ReLU
    return nn.Hardswish if config.use_hs else nn.ReLU

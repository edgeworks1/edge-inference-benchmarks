"""MobileNetV3-Small for BLE IQ data represented as a small 2D image.

One sample forms a single-channel image with shape ``4 x 75``. The four rows
are local I, local Q, remote I and remote Q; the 75 columns are IQ sample
positions. Flat inputs from the existing pipeline are reshaped internally.
"""

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn


BlockActivation = Literal["hardswish", "relu"]


def _make_divisible(value: float, divisor: int = 8) -> int:
    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )


class SqueezeExcitation2D(nn.Module):
    def __init__(self, channels: int, squeeze_channels: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, kernel_size=1)
        self.scale_activation = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return x * self.scale_activation(scale)


@dataclass(frozen=True)
class InvertedResidualConfig:
    kernel: int
    expanded_channels: int
    out_channels: int
    use_se: bool
    use_hs: bool
    stride: int


class InvertedResidual2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        config: InvertedResidualConfig,
        block_activation: BlockActivation = "hardswish",
    ) -> None:
        super().__init__()
        if config.stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        # "hardswish": Original MobileNetV3 (HardSwish nur in Blöcken mit use_hs).
        # "relu": ReLU in allen Blöcken, z.B. weil TFLM HardSwish nicht fusen kann.
        if block_activation == "hardswish":
            activation = nn.Hardswish if config.use_hs else nn.ReLU
        else:
            activation = nn.ReLU
        layers: list[nn.Module] = []
        if config.expanded_channels != in_channels:
            layers.append(
                ConvNormActivation(
                    in_channels, config.expanded_channels, 1, activation=activation
                )
            )
        layers.append(
            ConvNormActivation(
                config.expanded_channels,
                config.expanded_channels,
                config.kernel,
                stride=config.stride,
                groups=config.expanded_channels,
                activation=activation,
            )
        )
        if config.use_se:
            layers.append(
                SqueezeExcitation2D(
                    config.expanded_channels,
                    _make_divisible(config.expanded_channels / 4),
                )
            )
        layers.append(
            nn.Sequential(
                nn.Conv2d(config.expanded_channels, config.out_channels, 1, bias=False),
                nn.BatchNorm2d(config.out_channels),
            )
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = config.stride == 1 and in_channels == config.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        return result + x if self.use_residual else result


class MobileNetV3Small2D(nn.Module):
    """MobileNetV3-Small with a regression head for 4 x 75 IQ images."""

    def __init__(
        self,
        image_height: int = 4,
        image_width: int = 75,
        num_outputs: int = 1,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        block_activation: BlockActivation = "hardswish",
    ) -> None:
        super().__init__()
        if width_mult <= 0:
            raise ValueError("width_mult must be positive")
        if block_activation not in ("hardswish", "relu"):
            raise ValueError("block_activation must be 'hardswish' or 'relu'")
        self.image_height = image_height
        self.image_width = image_width

        def channels(value: int) -> int:
            return _make_divisible(value * width_mult)

        raw_config: Sequence[tuple[int, int, int, bool, bool, int]] = (
            (3, 16, 16, True, False, 2),
            (3, 72, 24, False, False, 2),
            (3, 88, 24, False, False, 1),
            (5, 96, 40, True, True, 2),
            (5, 240, 40, True, True, 1),
            (5, 240, 40, True, True, 1),
            (5, 120, 48, True, True, 1),
            (5, 144, 48, True, True, 1),
            (5, 288, 96, True, True, 2),
            (5, 576, 96, True, True, 1),
            (5, 576, 96, True, True, 1),
        )
        configs = [
            InvertedResidualConfig(k, channels(exp), channels(out), se, hs, stride)
            for k, exp, out, se, hs, stride in raw_config
        ]

        first_channels = channels(16)
        layers: list[nn.Module] = [
            ConvNormActivation(1, first_channels, 3, stride=2, activation=nn.Hardswish)
        ]
        current_channels = first_channels
        for config in configs:
            layers.append(InvertedResidual2D(current_channels, config, block_activation))
            current_channels = config.out_channels

        last_conv_channels = channels(576)
        layers.append(
            ConvNormActivation(
                current_channels, last_conv_channels, 1, activation=nn.Hardswish
            )
        )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        last_hidden_channels = channels(1024)
        self.head = nn.Sequential(
            nn.Linear(last_conv_channels, last_hidden_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(last_hidden_channels, num_outputs),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
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

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            expected = self.image_height * self.image_width
            if x.shape[1] != expected:
                raise ValueError(f"Expected {expected} flat features, got {x.shape[1]}")
            x = x.reshape(x.shape[0], 1, self.image_height, self.image_width)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(
                "Input must have shape (batch, 300), (batch, 4, 75), "
                "or (batch, 1, 4, 75)"
            )

        expected_shape = (1, self.image_height, self.image_width)
        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected shape (batch, {expected_shape[0]}, "
                f"{expected_shape[1]}, {expected_shape[2]}), got {tuple(x.shape)}"
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(torch.flatten(x, 1))


def mobilenet_v3_small_2d(**kwargs: object) -> MobileNetV3Small2D:
    return MobileNetV3Small2D(**kwargs)

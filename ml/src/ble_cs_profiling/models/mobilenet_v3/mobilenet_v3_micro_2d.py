"""MobileNetV3-Micro for BLE IQ data (4 x 75).

Optimized for nRF5340 internal SRAM (< 400 KB total size post-quantization,
fits entirely into SRAM for zero-wait-state execution).
"""

from dataclasses import dataclass
from typing import Sequence
import torch
from torch import nn


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
        self, in_channels: int, config: InvertedResidualConfig
    ) -> None:
        super().__init__()
        if config.stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        activation = nn.Hardswish if config.use_hs else nn.ReLU
        layers: list[nn.Module] = []

        # 1. Pointwise expansion
        if config.expanded_channels != in_channels:
            layers.append(
                ConvNormActivation(
                    in_channels,
                    config.expanded_channels,
                    1,
                    activation=activation,
                )
            )

        # 2. Depthwise convolution
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

        # 3. Squeeze & Excitation
        if config.use_se:
            squeeze_ch = max(4, _make_divisible(config.expanded_channels / 4))
            layers.append(SqueezeExcitation2D(config.expanded_channels, squeeze_ch))

        # 4. Pointwise linear projection
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    config.expanded_channels, config.out_channels, 1, bias=False
                ),
                nn.BatchNorm2d(config.out_channels),
            )
        )

        self.block = nn.Sequential(*layers)
        self.use_residual = (
            config.stride == 1 and in_channels == config.out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        return result + x if self.use_residual else result


class MobileNetV3Micro2D(nn.Module):
    """Kompakte Variante für Cortex-M33 (nRF5340 SRAM Residenz)."""

    def __init__(
        self,
        image_height: int = 4,
        image_width: int = 75,
        num_outputs: int = 1,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width

        def ch(value: int) -> int:
            return _make_divisible(value * width_mult)

        # Schlankes Backbone:
        # Format: (kernel, exp_ch, out_ch, use_se, use_hs, stride)
        # H x W Verlauf: (4x75) -> Stride 2 -> (2x38) -> Stride 2 -> (1x19) -> Rest Stride 1
        raw_config: Sequence[tuple[int, int, int, bool, bool, int]] = (
            (3, 16, 16, True, False, 2),  # -> 2 x 38
            (3, 48, 24, False, False, 2),  # -> 1 x 19 (Höhe ist nun 1)
            (3, 72, 24, False, False, 1),
            (5, 72, 40, True, True, 1),  # Größerer Kernel entlang der 19 Samples
            (5, 120, 40, True, True, 1),
            (5, 120, 48, True, True, 1),
            (5, 144, 48, True, True, 1),
            (5, 192, 64, True, True, 1),
        )

        configs = [
            InvertedResidualConfig(k, ch(exp), ch(out), se, hs, stride)
            for k, exp, out, se, hs, stride in raw_config
        ]

        first_channels = ch(16)
        layers: list[nn.Module] = [
            ConvNormActivation(
                1, first_channels, 3, stride=1, activation=nn.Hardswish
            )
        ]

        current_channels = first_channels
        for config in configs:
            layers.append(InvertedResidual2D(current_channels, config))
            current_channels = config.out_channels

        # Letzter 1x1 Conv vor dem Pooling: gedeckelt auf 96 statt 576
        last_conv_channels = ch(96)
        layers.append(
            ConvNormActivation(
                current_channels, last_conv_channels, 1, activation=nn.Hardswish
            )
        )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Head drastisch verkleinert: 96 -> 64 -> num_outputs (statt 576 -> 1024 -> 1)
        last_hidden_channels = ch(64)
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
                raise ValueError(
                    f"Expected {expected} flat features, got {x.shape[1]}"
                )
            x = x.reshape(x.shape[0], 1, self.image_height, self.image_width)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(
                "Input must have shape (batch, 300), (batch, 4, 75), "
                "or (batch, 1, 4, 75)"
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_input(x)
        x = self.features(x)
        x = self.avgpool(x)
        return self.head(torch.flatten(x, 1))


def mobilenet_v3_micro_2d(**kwargs: object) -> MobileNetV3Micro2D:
    return MobileNetV3Micro2D(**kwargs)
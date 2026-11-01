"""MobileNetV3-Small with deliberately misaligned channel counts, to measure SIMD penalties.

Almost the same parameter count as MobileNetV3-Small, but every channel count is odd
(one less than the original), so no layer is divisible by 4 or 8. CMSIS-NN kernels then
have to process the leftover channels outside their unrolled 4-channel SIMD loops.
"""

from edge_ib.models.mobilenet_v3.blocks import (
    BlockRow,
    MobileNetV3Regressor2D,
    block_configs,
)

# (kernel, expanded_channels, out_channels, use_se, use_hs, stride); original values in comments
BLOCKS: tuple[BlockRow, ...] = (
    (3, 15, 15, True, False, 2),  # 16, 16
    (3, 71, 23, False, False, 2),  # 72, 24
    (3, 87, 23, False, False, 1),  # 88, 24
    (5, 95, 39, True, True, 2),  # 96, 40
    (5, 239, 39, True, True, 1),  # 240, 40
    (5, 239, 39, True, True, 1),  # 240, 40
    (5, 119, 47, True, True, 1),  # 120, 48
    (5, 143, 47, True, True, 1),  # 144, 48
    (5, 287, 95, True, True, 2),  # 288, 96
    (5, 575, 95, True, True, 1),  # 576, 96
    (5, 575, 95, True, True, 1),  # 576, 96
)


def odd_squeeze_channels(expanded_channels: int) -> int:
    """A quarter of the channels, forced odd (``| 1`` sets the lowest bit)."""
    return max(1, (expanded_channels // 4) | 1)


class MobileNetV3Small2DBadSimd(MobileNetV3Regressor2D):
    """MobileNetV3-Small with odd channel counts in every layer."""

    def __init__(
        self,
        image_height: int = 4,
        image_width: int = 75,
        num_outputs: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            blocks=block_configs(BLOCKS),
            stem_channels=15,  # 16
            stem_stride=2,
            last_conv_channels=575,  # 576
            hidden_channels=1023,  # 1024
            dropout=dropout,
            image_height=image_height,
            image_width=image_width,
            num_outputs=num_outputs,
            squeeze_channels=odd_squeeze_channels,
        )

"""MobileNetV3-Micro: a slimmed-down variant that fits the nRF5340 (about 345 KiB as INT8).

Compared to MobileNetV3-Small: no stride in the stem, fewer and narrower blocks, and a much
smaller last conv and head (96 -> 64 instead of 576 -> 1024).
"""

from ble_cs_profiling.models.mobilenet_v3.blocks import BlockRow, MobileNetV3Regressor2D, block_configs, make_divisible

# (kernel, expanded_channels, out_channels, use_se, use_hs, stride)
# Height x width: 4 x 75 -> stride 2 -> 2 x 38 -> stride 2 -> 1 x 19, then stride 1
BLOCKS: tuple[BlockRow, ...] = (
    (3, 16, 16, True, False, 2),
    (3, 48, 24, False, False, 2),
    (3, 72, 24, False, False, 1),
    (5, 72, 40, True, True, 1),  # larger kernel along the 19 remaining sample positions
    (5, 120, 40, True, True, 1),
    (5, 120, 48, True, True, 1),
    (5, 144, 48, True, True, 1),
    (5, 192, 64, True, True, 1),
)


class MobileNetV3Micro2D(MobileNetV3Regressor2D):
    """Compact MobileNetV3 with a regression head for the Cortex-M33."""

    def __init__(
        self,
        image_height: int = 4,
        image_width: int = 75,
        num_outputs: int = 1,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        def scale(channels: int) -> int:
            return make_divisible(channels * width_mult)

        super().__init__(
            blocks=block_configs(BLOCKS, scale),
            stem_channels=scale(16),
            stem_stride=1,
            last_conv_channels=scale(96),
            hidden_channels=scale(64),
            dropout=dropout,
            image_height=image_height,
            image_width=image_width,
            num_outputs=num_outputs,
        )

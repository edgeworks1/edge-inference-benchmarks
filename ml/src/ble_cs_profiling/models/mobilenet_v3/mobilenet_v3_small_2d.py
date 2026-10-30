"""MobileNetV3-Small (original block table) for 4 x 75 BLE IQ images.

Too large for the nRF5340 (about 1.9 MB as INT8), runs in Renode and serves as reference.
"""

from ble_cs_profiling.models.mobilenet_v3.blocks import (
    BlockActivation,
    BlockRow,
    MobileNetV3Regressor2D,
    block_configs,
    make_divisible,
)

# (kernel, expanded_channels, out_channels, use_se, use_hs, stride), as in the MobileNetV3 paper
BLOCKS: tuple[BlockRow, ...] = (
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


class MobileNetV3Small2D(MobileNetV3Regressor2D):
    """MobileNetV3-Small with a regression head.

    ``block_activation="relu"`` replaces HardSwish in all blocks, e.g. because TFLite Micro
    cannot fuse HardSwish into the preceding convolution.
    """

    def __init__(
        self,
        image_height: int = 4,
        image_width: int = 75,
        num_outputs: int = 1,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        block_activation: BlockActivation = "hardswish",
    ) -> None:
        if width_mult <= 0:
            raise ValueError("width_mult must be positive")

        def scale(channels: int) -> int:
            return make_divisible(channels * width_mult)

        super().__init__(
            blocks=block_configs(BLOCKS, scale),
            stem_channels=scale(16),
            stem_stride=2,
            last_conv_channels=scale(576),
            hidden_channels=scale(1024),
            dropout=dropout,
            image_height=image_height,
            image_width=image_width,
            num_outputs=num_outputs,
            block_activation=block_activation,
        )

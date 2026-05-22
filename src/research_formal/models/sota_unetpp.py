"""SOTA-style retinal vessel segmentation baseline.

This baseline uses segmentation-models-pytorch U-Net++ with a pretrained encoder.
It is intentionally dependency-light at import time: the optional SMP dependency is
imported only when the model is built, which keeps the vanilla CMUNeXt path usable.
"""

from __future__ import annotations

from torch import nn


def build_sota_unetpp(
    encoder_name: str = "efficientnet-b0",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "The SOTA pipeline requires segmentation-models-pytorch. "
            "Install it with: pip install segmentation-models-pytorch"
        ) from exc

    return smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )

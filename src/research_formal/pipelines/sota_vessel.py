"""SOTA-style thin-vessel segmentation inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research_formal.models.sota_unetpp import build_sota_unetpp


@dataclass(frozen=True)
class SOTAVesselConfig:
    encoder_name: str = "efficientnet-b0"
    encoder_weights: str | None = "imagenet"
    image_size: int = 512
    input_channels: int = 3
    num_classes: int = 1
    device: str = "cpu"


class SOTAVesselPipeline:
    def __init__(self, config: SOTAVesselConfig | None = None) -> None:
        self.config = config or SOTAVesselConfig()
        self.device = torch.device(self.config.device)
        self.model = build_sota_unetpp(
            encoder_name=self.config.encoder_name,
            encoder_weights=self.config.encoder_weights,
            in_channels=self.config.input_channels,
            classes=self.config.num_classes,
        ).to(self.device)
        self.model.eval()

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)

    def preprocess_image(self, image_path: str | Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.image_size, self.config.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.model(image_tensor.to(self.device))
        return torch.sigmoid(logits)

    def predict_image(self, image_path: str | Path) -> torch.Tensor:
        return self.predict_tensor(self.preprocess_image(image_path))

    def save_probability_mask(self, probabilities: torch.Tensor, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask = probabilities.detach().cpu().squeeze().clamp(0.0, 1.0).numpy()
        image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        image.save(output_path)

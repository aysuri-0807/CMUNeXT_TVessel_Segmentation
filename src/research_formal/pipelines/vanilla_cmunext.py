"""Untrained vanilla CMUNeXt inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from research_formal.models.cmunext import cmunext, cmunext_l, cmunext_s, count_trainable_parameters


MODEL_FACTORIES = {
    "cmunext": cmunext,
    "cmunext_s": cmunext_s,
    "cmunext_l": cmunext_l,
}


@dataclass(frozen=True)
class VanillaCMUNeXtConfig:
    variant: str = "cmunext"
    image_size: int = 256
    input_channels: int = 3
    num_classes: int = 1
    device: str = "cpu"


class VanillaCMUNeXtPipeline:
    def __init__(self, config: VanillaCMUNeXtConfig | None = None) -> None:
        self.config = config or VanillaCMUNeXtConfig()
        if self.config.variant not in MODEL_FACTORIES:
            valid = ", ".join(sorted(MODEL_FACTORIES))
            raise ValueError(f"Unknown CMUNeXt variant '{self.config.variant}'. Valid variants: {valid}")

        self.device = torch.device(self.config.device)
        self.model = MODEL_FACTORIES[self.config.variant](
            input_channel=self.config.input_channels,
            num_classes=self.config.num_classes,
        ).to(self.device)
        self.model.eval()

    @property
    def trainable_parameters(self) -> int:
        return count_trainable_parameters(self.model)

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

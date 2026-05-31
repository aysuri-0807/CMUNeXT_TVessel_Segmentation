from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_formal.pipelines.sota_vessel import SOTAVesselConfig, SOTAVesselPipeline
from research_formal.pipelines.vanilla_cmunext import VanillaCMUNeXtConfig, VanillaCMUNeXtPipeline


DEFAULT_WEIGHTS = {
    "vanilla": ROOT / "vanilla_training" / "best_vanilla_cmunext_l.pt",
    "sota": ROOT / "sota_training" / "best_sota_unetpp.pt",
}


VANILLA_VARIANTS = {
    "cmunext": "cmunext",
    "cmunext-s": "cmunext_s",
    "cmunext_s": "cmunext_s",
    "cmunext-l": "cmunext_l",
    "cmunext_l": "cmunext_l",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trained retinal vessel segmentation with the vanilla or SOTA model."
    )
    parser.add_argument("--model", choices=["vanilla", "sota"], required=True, help="Model family to run.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the image to segment.")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Optional checkpoint path. Defaults to the matching training folder checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output filename/path. Relative paths are written under root/output.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        help="Optional square inference size. Defaults to the checkpoint config or 512.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for a binary mask. Use --probability-mask to save raw probabilities instead.",
    )
    parser.add_argument(
        "--probability-mask",
        action="store_true",
        help="Save a grayscale probability mask instead of a thresholded binary mask.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, for example cpu or cuda.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    config: dict[str, Any] = {}

    if isinstance(checkpoint, dict):
        raw_config = checkpoint.get("config", {})
        if isinstance(raw_config, dict):
            config = raw_config

        for key in ("model_state_dict", "state_dict", "model", "net"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return clean_state_dict(state_dict), config

        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return clean_state_dict(checkpoint), config

    raise ValueError(f"Could not find a model state dict inside checkpoint: {path}")


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "model."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned[clean_key] = value
    return cleaned


def normalize_vanilla_variant(value: Any) -> str:
    variant = str(value or "cmunext_l").strip().lower()
    variant = variant.replace(" ", "_")
    return VANILLA_VARIANTS.get(variant, VANILLA_VARIANTS.get(variant.replace("_", "-"), "cmunext_l"))


def build_pipeline(
    model_name: str,
    checkpoint_config: dict[str, Any],
    image_size: int,
    device: torch.device,
) -> VanillaCMUNeXtPipeline | SOTAVesselPipeline:
    if model_name == "vanilla":
        variant = normalize_vanilla_variant(checkpoint_config.get("variant"))
        return VanillaCMUNeXtPipeline(
            VanillaCMUNeXtConfig(
                variant=variant,
                image_size=image_size,
                device=str(device),
            )
        )

    encoder_name = str(checkpoint_config.get("encoder_name") or "efficientnet-b0")
    return SOTAVesselPipeline(
        SOTAVesselConfig(
            encoder_name=encoder_name,
            encoder_weights=None,
            image_size=image_size,
            device=str(device),
        )
    )


def resolve_output_path(image_path: Path, model_name: str, output: Path | None) -> Path:
    if output is None:
        output_path = ROOT / "output" / f"{image_path.stem}_{model_name}_mask.png"
    elif output.is_absolute():
        output_path = output
    else:
        output_path = ROOT / "output" / output

    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    return output_path


def save_mask(
    probabilities: torch.Tensor,
    output_path: Path,
    original_size: tuple[int, int],
    threshold: float | None,
) -> None:
    mask = probabilities.detach().cpu().squeeze().clamp(0.0, 1.0).numpy()
    if threshold is None:
        mask_array = (mask * 255).astype(np.uint8)
        resample = Image.BILINEAR
    else:
        mask_array = (mask >= threshold).astype(np.uint8) * 255
        resample = Image.NEAREST

    mask_image = Image.fromarray(mask_array, mode="L")
    if mask_image.size != original_size:
        mask_image = mask_image.resize(original_size, resample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_image.save(output_path)


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    if args.threshold < 0.0 or args.threshold > 1.0:
        raise SystemExit("--threshold must be between 0.0 and 1.0.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")

    weights_path = (args.weights or DEFAULT_WEIGHTS[args.model]).resolve()
    state_dict, checkpoint_config = load_checkpoint(weights_path, device)

    checkpoint_image_size = checkpoint_config.get("image_size", 512)
    image_size = args.image_size or int(checkpoint_image_size)
    pipeline = build_pipeline(args.model, checkpoint_config, image_size, device)
    pipeline.model.load_state_dict(state_dict, strict=True)
    pipeline.model.eval()

    with Image.open(image_path) as image:
        original_size = image.size

    probabilities = pipeline.predict_image(image_path)
    output_path = resolve_output_path(image_path, args.model, args.output)
    threshold = None if args.probability_mask else args.threshold
    save_mask(probabilities, output_path, original_size, threshold)

    print(f"model={args.model}")
    print(f"weights={weights_path}")
    print(f"device={pipeline.device}")
    print(f"image_size={image_size}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from research_formal.pipelines.sota_vessel import SOTAVesselConfig, SOTAVesselPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SOTA-style U-Net++ vessel segmentation pipeline.")
    parser.add_argument("--image", type=Path, help="Input image path.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sota_vessel_mask.png")
    parser.add_argument("--encoder", default="efficientnet-b0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Run a random-tensor smoke test instead of image inference.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SOTAVesselConfig(
        encoder_name=args.encoder,
        image_size=args.image_size,
        device=args.device,
    )
    pipeline = SOTAVesselPipeline(config)

    if args.smoke:
        tensor = torch.rand(1, 3, args.image_size, args.image_size, device=pipeline.device)
        probabilities = pipeline.predict_tensor(tensor)
        print(f"encoder={args.encoder}")
        print(f"device={pipeline.device}")
        print(f"trainable_parameters={pipeline.trainable_parameters}")
        print(f"output_shape={tuple(probabilities.shape)}")
        print(f"probability_range=({probabilities.min().item():.4f}, {probabilities.max().item():.4f})")
        return

    if args.image is None:
        raise SystemExit("--image is required unless --smoke is used.")

    probabilities = pipeline.predict_image(args.image)
    pipeline.save_probability_mask(probabilities, args.output)
    print(f"encoder={args.encoder}")
    print(f"device={pipeline.device}")
    print(f"trainable_parameters={pipeline.trainable_parameters}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()

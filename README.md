# Research Formal

This workspace contains research code for evaluating CMUNeXt on retinal thin-vessel segmentation.

## Step 1: Vanilla CMUNeXt Pipeline

The first baseline is an untrained vanilla CMUNeXt segmentation pipeline. It includes:

- A PyTorch implementation of the official CMUNeXt architecture variants.
- A CLI that loads an image, runs an untrained forward pass, and writes a probability mask.
- A smoke test that verifies the model produces an output mask with the expected shape.

Run an untrained inference pass after installing dependencies:

```powershell
python scripts/run_vanilla_cmunext.py --image path\to\fundus.png --output outputs\vanilla_mask.png
```

Run the smoke test:

```powershell
python scripts/run_vanilla_cmunext.py --smoke
```

Available model variants:

- `cmunext`
- `cmunext_s`
- `cmunext_l`

## Step 2: Colab Training Notebooks

Training notebooks are organized by experiment:

- `vanilla_training/vanilla_cmunext_training_colab.ipynb` trains the unmodified CMUNeXt-L baseline from `FengheTan9/CMUNeXt`.
- `sota_training/sota_vessel_training_colab.ipynb` trains a SOTA-style U-Net++ EfficientNet vessel baseline.

Both notebooks expect a dataset folder with matching image and mask filenames:

```text
DATA_ROOT/
  images/
    sample_001.png
  masks/
    sample_001.png
```

Both notebooks use the Kaggle dataset `ipythonx/retinal-vessel-segmentation`. Edit `PROJECT_ROOT` and `DATASET_SUBDIR` at the top of each notebook if needed before running in Colab. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets so the notebooks can pull the dataset through the Kaggle API.

Each notebook writes per-epoch metrics to `training_history.csv` in its output folder during training, so partial runs still leave usable logs. Use `observations.md` as the running research log for training behavior, qualitative notes, and comparison conclusions.

## SOTA-Style Pipeline

The SOTA comparison pipeline uses `segmentation-models-pytorch` U-Net++ with a pretrained EfficientNet encoder and vessel-aware loss configuration in the Colab notebook. Local inference entrypoint:

```powershell
python scripts/run_sota_vessel.py --image path\to\fundus.png --output outputs\sota_mask.png
```

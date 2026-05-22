# Vanilla CMUNeXt-L Colab Training

Open `vanilla_cmunext_training_colab.ipynb` in Google Colab and edit:

- `PROJECT_ROOT`
- `KAGGLE_DATASET`
- `DATASET_SUBDIR`
- `IMAGE_SIZE`
- `BATCH_SIZE`
- `EPOCHS`

Before running, add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets. The notebook pulls `ipythonx/retinal-vessel-segmentation` through the Kaggle API and expects the extracted dataset to contain `images/` and `masks/` folders, or those folders under `DATASET_SUBDIR`.

The notebook trains the unmodified CMUNeXt-L baseline using the large configuration from `FengheTan9/CMUNeXt` and saves:

- `outputs/vanilla_cmunext/best_vanilla_cmunext_l.pt`
- `outputs/vanilla_cmunext/training_history.csv`

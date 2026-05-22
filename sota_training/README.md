# SOTA-Style Vessel Colab Training

Open `sota_vessel_training_colab.ipynb` in Google Colab and edit:

- `PROJECT_ROOT`
- `KAGGLE_DATASET`
- `DATASET_SUBDIR`
- `IMAGE_SIZE`
- `BATCH_SIZE`
- `EPOCHS`
- `ENCODER_NAME`

Before running, add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets. The notebook pulls `ipythonx/retinal-vessel-segmentation` through the Kaggle API and expects the extracted dataset to contain `images/` and `masks/` folders, or those folders under `DATASET_SUBDIR`.

The notebook trains a U-Net++ EfficientNet baseline for thin retinal vessel segmentation and saves:

- `outputs/sota_unetpp/best_sota_unetpp.pt`
- `outputs/sota_unetpp/training_history.csv`

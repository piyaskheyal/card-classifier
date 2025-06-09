import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Set your target folder
target_dir = "/home/kheyal/dev/card-classifier/data"
os.makedirs(target_dir, exist_ok=True)

# Download dataset
api.dataset_download_files(
    "gpiosenka/cards-image-datasetclassification",
    path=target_dir,
    unzip=True
)

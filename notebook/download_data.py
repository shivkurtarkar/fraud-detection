import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Set the path for downloading
download_path = "../data"

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Dataset details
dataset = "kartik2112/fraud-detection"

# Download dataset to the specified path
api.dataset_download_files(dataset, path=download_path, unzip=True)

print(f"Dataset downloaded to: {download_path}")

import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(dataset, download_path='./data'):
    # Initialize API
    api = KaggleApi()
    api.authenticate()

    # Create download path if not exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Download dataset and unzip
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print(f'Dataset downloaded and extracted to {download_path}')


if __name__ == "__main__":
    download_kaggle_dataset('emmarex/plantdisease')

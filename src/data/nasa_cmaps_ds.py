import logging

from lib.data.kaggle import download_dataset, unzip_file
from src.definitions import EXTERNAL_DATA_FOLDER


def download_nasa_ds(
    logger: logging.Logger = logging.getLogger(__name__),
):
    archive = download_dataset(
        "behrad3d",
        "nasa-cmaps",
        EXTERNAL_DATA_FOLDER,
        logger,
    )

    return unzip_file(archive, logger)

import logging
from pathlib import Path
import zipfile
import kaggle


def download_dataset(owner: str, name: str, dest: Path, logger: logging.Logger) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    file_path = dest / f"{name}.zip"

    if file_path.is_file():
        logger.info(
            "Found [ %s ] dataset in [ %s ]. Skipping download...", name, file_path
        )
    else:
        logger.info("Downloading [ %s ] dataset to [ %s ]", name, file_path)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=f"{owner}/{name}", path=dest)

    return file_path


def unzip_file(archive: Path, logger: logging.Logger) -> Path:
    archive_name = archive.stem
    dest_file = archive.parent / archive_name

    if dest_file.exists():
        logger.info("[ %s ] is already unzipped. Skipping ...", dest_file)
    else:
        logger.info("Unzipping [ %s ] to [ %s ]", archive, dest_file)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(dest_file)

    return dest_file

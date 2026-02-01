"""Download and prepare DSTC2 dataset."""

import json
import logging
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# DSTC2 data URLs
DSTC2_URLS = {
    "train": "https://github.com/matthen/dstc/raw/master/data/dstc2_traindev.zip",
    "test": "https://github.com/matthen/dstc/raw/master/data/dstc2_test.zip",
}

DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file."""
    logger.info(f"Extracting {zip_path} to {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def download_dstc2(data_dir: Path | None = None, force: bool = False) -> Path:
    """
    Download and extract DSTC2 dataset.

    Args:
        data_dir: Directory to store data. Defaults to project data/ directory.
        force: If True, re-download even if files exist.

    Returns:
        Path to the raw data directory.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, url in DSTC2_URLS.items():
        zip_path = raw_dir / f"dstc2_{name}.zip"
        extract_marker = raw_dir / f".{name}_extracted"

        # Download if needed
        if not zip_path.exists() or force:
            logger.info(f"Downloading {name} data from {url}")
            download_file(url, zip_path)
        else:
            logger.info(f"{name} data already downloaded")

        # Extract if needed
        if not extract_marker.exists() or force:
            extract_zip(zip_path, raw_dir)
            extract_marker.touch()
        else:
            logger.info(f"{name} data already extracted")

    logger.info(f"DSTC2 data ready at {raw_dir}")
    return raw_dir


def main() -> None:
    """CLI entry point for downloading data."""
    logging.basicConfig(level=logging.INFO)
    download_dstc2()


if __name__ == "__main__":
    main()

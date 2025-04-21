import os
import zipfile
import cv2
from pathlib import Path
import gdown
from src import logger
from src.utils.common import get_size, create_directories, add_noise
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """Initialize DataIngestion with configuration."""
        self.config = config
        create_directories([self.config.root_dir, self.config.noisy_dir])

    def download_file(self):
        """Download the zip file from Google Drive."""
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            if not zip_download_dir.exists():
                logger.info(f"Downloading data from {dataset_url} to {zip_download_dir}")
                file_id = dataset_url.split("/")[5]
                prefix = 'https://drive.google.com/uc?export=download&id='
                gdown.download(prefix + file_id, str(zip_download_dir), quiet=False)
                logger.info(f"Downloaded data to {zip_download_dir}: {get_size(zip_download_dir)}")
            else:
                logger.info(f"Dataset already exists at {zip_download_dir}: {get_size(zip_download_dir)}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise e

    def extract_zip_file(self):
        """Extract the downloaded zip file."""
        try:
            unzip_path = self.config.unzip_dir
            create_directories([unzip_path])
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted dataset to {unzip_path}")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise e

    def generate_noisy_images(self):
        """Generate noisy versions of images."""
        try:
            noise_type = self.config.params.get("NOISE_TYPE", "gaussian")
            noise_factor = self.config.params.get("NOISE_FACTOR", 0.1)
            data_dir = self.config.unzip_dir
            noisy_data_dir = self.config.noisy_dir
            for label in ['NORMAL', 'PNEUMONIA']:
                label_dir = data_dir / "pneumonia_xray" / label
                noisy_label_dir = noisy_data_dir / label
                create_directories([noisy_label_dir])
                if not label_dir.exists():
                    logger.warning(f"Directory {label_dir} not found")
                    continue
                for img_path in label_dir.glob('*.jpeg'):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        logger.warning(f"Failed to load image: {img_path}")
                        continue
                    img = cv2.resize(img, tuple(self.config.params["IMAGE_SIZE"])) / 255.0
                    noisy_img = add_noise(img, noise_type=noise_type, noise_factor=noise_factor)
                    cv2.imwrite(str(noisy_label_dir / img_path.name), noisy_img * 255)
                logger.info(f"Generated noisy images for {label} at {noisy_label_dir}")
        except Exception as e:
            logger.error(f"Noisy image generation failed: {e}")
            raise e
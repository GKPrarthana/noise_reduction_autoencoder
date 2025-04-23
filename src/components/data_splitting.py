from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
from src import logger
from src.utils.common import create_directories
from src.entity.config_entity import DataSplittingConfig

class DataSplitting:
    def __init__(self, config: DataSplittingConfig):
        """Initialize with configuration."""
        self.config = config
        create_directories([
            self.config.train_clean_dir, self.config.train_noisy_dir,
            self.config.val_clean_dir, self.config.val_noisy_dir,
            self.config.test_clean_dir, self.config.test_noisy_dir
        ])

    def split_dataset(self):
        """Split clean and noisy images into train, validation, and test sets."""
        try:
            logger.info("Starting data splitting")
            clean_base_dir = Path(self.config.clean_data_source)  # artifacts/data_ingestion/pneumonia_xray
            noisy_base_dir = Path(self.config.noisy_data_source)  # artifacts/data_ingestion/noisy_images
            train_ratio, val_ratio, test_ratio = self.config.split_ratios

            for label in ["NORMAL", "PNEUMONIA"]:
                # Collect all clean images directly from pneumonia_xray/NORMAL or PNEUMONIA
                label_clean_dir = clean_base_dir / label
                if not label_clean_dir.exists():
                    logger.warning(f"Directory {label_clean_dir} not found")
                    continue
                clean_images = list(label_clean_dir.glob("*.jpeg"))
                
                if not clean_images:
                    logger.warning(f"No clean images found for {label} in {clean_base_dir}")
                    continue

                # Map clean images to noisy images
                noisy_images = []
                for clean_img in clean_images:
                    noisy_img_path = noisy_base_dir / label / clean_img.name
                    if not noisy_img_path.exists():
                        logger.warning(f"Noisy image {noisy_img_path} not found for {clean_img}")
                        continue
                    noisy_images.append(noisy_img_path)

                # Ensure we have matching pairs
                if len(clean_images) != len(noisy_images):
                    logger.warning(f"Mismatch: {len(clean_images)} clean images vs {len(noisy_images)} noisy images for {label}")
                    continue

                # Split images
                train_clean, temp_clean = train_test_split(clean_images, train_size=train_ratio, random_state=42)
                val_clean, test_clean = train_test_split(temp_clean, train_size=val_ratio/(val_ratio + test_ratio), random_state=42)

                # Split noisy images using the same indices
                train_indices = [clean_images.index(img) for img in train_clean]
                val_indices = [clean_images.index(img) for img in val_clean]
                test_indices = [clean_images.index(img) for img in test_clean]
                train_noisy = [noisy_images[i] for i in train_indices]
                val_noisy = [noisy_images[i] for i in val_indices]
                test_noisy = [noisy_images[i] for i in test_indices]

                # Copy images to respective directories
                for split, clean_imgs, noisy_imgs, clean_dir, noisy_dir in [
                    ("train", train_clean, train_noisy, self.config.train_clean_dir / label, self.config.train_noisy_dir / label),
                    ("val", val_clean, val_noisy, self.config.val_clean_dir / label, self.config.val_noisy_dir / label),
                    ("test", test_clean, test_noisy, self.config.test_clean_dir / label, self.config.test_noisy_dir / label)
                ]:
                    create_directories([clean_dir, noisy_dir])
                    for clean_img, noisy_img in zip(clean_imgs, noisy_imgs):
                        shutil.copy(clean_img, clean_dir / clean_img.name)
                        shutil.copy(noisy_img, noisy_dir / noisy_img.name)
                    logger.info(f"Copied {len(clean_imgs)} images to {split} for {label}")
            logger.info("Data splitting completed")
        except Exception as e:
            logger.error(f"Data splitting failed: {e}", exc_info=True)
            raise e
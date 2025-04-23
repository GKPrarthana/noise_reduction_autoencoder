#update components
import tensorflow as tf
from pathlib import Path
import numpy as np
from src import logger
from src.entity.config_entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def load_and_preprocess_image(self, clean_path, noisy_path):
        """Load and preprocess a pair of clean and noisy images."""
        # Read images
        clean_img = tf.io.read_file(clean_path)
        noisy_img = tf.io.read_file(noisy_path)
        
        # Decode JPEG images
        clean_img = tf.image.decode_jpeg(clean_img, channels=1)  # Grayscale
        noisy_img = tf.image.decode_jpeg(noisy_img, channels=1)
        
        # Convert to float32 and normalize to [0, 1]
        clean_img = tf.cast(clean_img, tf.float32) / 255.0
        noisy_img = tf.cast(noisy_img, tf.float32) / 255.0
        
        # Resize images
        clean_img = tf.image.resize(clean_img, [self.config.image_size, self.config.image_size])
        noisy_img = tf.image.resize(noisy_img, [self.config.image_size, self.config.image_size])
        
        return noisy_img, clean_img  # Input: noisy, Target: clean

    def create_dataset(self, clean_paths, noisy_paths, split):
        """Create a TensorFlow dataset for a given split."""
        dataset = tf.data.Dataset.from_tensor_slices((clean_paths, noisy_paths))
        dataset = dataset.map(
            lambda clean, noisy: self.load_and_preprocess_image(clean, noisy),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply batching and shuffling for training set
        if split == "train":
            dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size)
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def preprocess(self):
        """Preprocess images and create datasets for train, val, and test splits."""
        try:
            logger.info("Starting data preprocessing")
            
            splits = ["train", "val", "test"]
            datasets = {}
            
            for split in splits:
                # Collect clean and noisy image paths
                clean_normal_dir = Path(self.config.data_dir) / split / "clean" / "NORMAL"
                clean_pneumonia_dir = Path(self.config.data_dir) / split / "clean" / "PNEUMONIA"
                noisy_normal_dir = Path(self.config.data_dir) / split / "noisy" / "NORMAL"
                noisy_pneumonia_dir = Path(self.config.data_dir) / split / "noisy" / "PNEUMONIA"
                
                # Get image paths
                clean_normal_paths = [str(p) for p in clean_normal_dir.glob("*.jpeg")]
                clean_pneumonia_paths = [str(p) for p in clean_pneumonia_dir.glob("*.jpeg")]
                noisy_normal_paths = [str(p) for p in noisy_normal_dir.glob("*.jpeg")]
                noisy_pneumonia_paths = [str(p) for p in noisy_pneumonia_dir.glob("*.jpeg")]
                
                # Combine paths
                clean_paths = clean_normal_paths + clean_pneumonia_paths
                noisy_paths = noisy_normal_paths + noisy_pneumonia_paths
                
                # Sort to ensure pairing
                clean_paths.sort()
                noisy_paths.sort()
                
                # Log counts
                logger.info(f"{split} set: {len(clean_paths)} clean images, {len(noisy_paths)} noisy images")
                
                if len(clean_paths) != len(noisy_paths):
                    raise ValueError(f"Mismatch in {split} set: {len(clean_paths)} clean vs {len(noisy_paths)} noisy images")
                
                if len(clean_paths) == 0:
                    raise ValueError(f"No images found in {split} set")
                
                # Create dataset
                dataset = self.create_dataset(clean_paths, noisy_paths, split)
                datasets[split] = dataset
            
            logger.info("Data preprocessing completed")
            return datasets
        
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}", exc_info=True)
            raise e
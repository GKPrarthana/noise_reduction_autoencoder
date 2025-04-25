from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    noisy_dir: Path
    params: dict
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params: dict
    
@dataclass(frozen=True)
class DataSplittingConfig:
    root_dir: Path
    train_clean_dir: Path
    train_noisy_dir: Path
    val_clean_dir: Path
    val_noisy_dir: Path
    test_clean_dir: Path
    test_noisy_dir: Path
    split_ratios: List[float]
    clean_data_source: Path
    noisy_data_source: Path
    params: dict
    
@dataclass(frozen=True)
class DataPreprocessingConfig:
    data_dir: Path
    image_size: int
    batch_size: int
    shuffle_buffer_size: int
    
@dataclass(frozen=True)
class ModelTrainingConfig:
    model_dir: Path
    image_size: int
    epochs: int
    
@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    test_data: dict  # Preprocessed test dataset
    all_params: dict
    mlflow_uri: str
    params_image_size: int
    params_batch_size: int
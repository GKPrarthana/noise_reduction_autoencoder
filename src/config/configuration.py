#update config manager
from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, DataSplittingConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir,config.noisy_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir) ,
            noisy_dir=Path(config.noisy_dir),
            params=self.params
        )
        
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([Path(config.root_dir)])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params=self.params
        )
        
    def get_data_splitting_config(self) -> DataSplittingConfig:
        config = self.config.data_splitting
        create_directories([
            Path(config.root_dir), Path(config.train_clean_dir), Path(config.train_noisy_dir),
            Path(config.val_clean_dir), Path(config.val_noisy_dir),
            Path(config.test_clean_dir), Path(config.test_noisy_dir)
        ])
        return DataSplittingConfig(
            root_dir=Path(config.root_dir),
            train_clean_dir=Path(config.train_clean_dir),
            train_noisy_dir=Path(config.train_noisy_dir),
            val_clean_dir=Path(config.val_clean_dir),
            val_noisy_dir=Path(config.val_noisy_dir),
            test_clean_dir=Path(config.test_clean_dir),
            test_noisy_dir=Path(config.test_noisy_dir),
            split_ratios=config.split_ratios,
            clean_data_source=Path(config.clean_data_source),
            noisy_data_source=Path(config.noisy_data_source),
            params=self.params
        )
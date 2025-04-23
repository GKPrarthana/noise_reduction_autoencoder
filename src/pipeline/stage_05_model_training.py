from src.config.configuration import ConfigurationManager
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_training import ModelTraining
from src import logger

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    def __init__(self):
        self.datasets = None

    def main(self):
        try:
            # Load the preprocessed datasets from Stage 04
            config_manager = ConfigurationManager()
            data_preprocessing_config = config_manager.get_data_preprocessing_config()
            data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
            self.datasets = data_preprocessing.preprocess()

            # Initialize and train the model
            model_training_config = config_manager.get_model_training_config()
            model_training = ModelTraining(config=model_training_config, datasets=self.datasets)
            history = model_training.train()

            return history

        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        history = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
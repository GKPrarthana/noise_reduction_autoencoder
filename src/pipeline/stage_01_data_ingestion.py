from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    """
    Pipeline class to handle data download, extraction,
    and noisy image generation.
    """
    def __init__(self):
        pass

    def run(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        data_ingestion.generate_noisy_images()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.run()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

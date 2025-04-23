#update pipeline
from src.config.configuration import ConfigurationManager
from src.components.data_splitting import DataSplitting
from src import logger

STAGE_NAME = "Data Splitting"

class DataSplittingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_splitting_config = config.get_data_splitting_config()
        data_splitting = DataSplitting(config=data_splitting_config)
        data_splitting.split_dataset()


if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataSplittingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
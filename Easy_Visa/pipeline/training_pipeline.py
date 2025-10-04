import sys
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from Easy_Visa.components.data_ingestion import DataIngestion

from Easy_Visa.entity.config_entity import DataIngestionConfig
from Easy_Visa.entity.config_entity import TrainingPipelineConfig
from Easy_Visa.entity.artifact_entity import DataIngestionArtifact


class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        self.data_ingestion_config=DataIngestionConfig(train_pipeline_config=self.training_pipeline_config)

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            logger.info("Entered the start_data_ingestion method")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initate_data_ingestion()
            logger.info("Got train test csv")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e)
        
    def run_pipeline(self)->None:
        try:
            data_ingestion_artifact=self.start_data_ingestion()
        except Exception as e:
            raise CustomException(e)
        
        


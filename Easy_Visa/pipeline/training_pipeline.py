import sys
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from Easy_Visa.components.data_ingestion import DataIngestion
from Easy_Visa.components.data_validation import DataValidation

from Easy_Visa.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig)
from Easy_Visa.entity.config_entity import TrainingPipelineConfig
from Easy_Visa.entity.artifact_entity import (DataIngestionArtifact,
                                              DataValidationArtifact)


class TrainPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
        self.data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)

    def start_data_ingestion(self)->DataIngestionArtifact:
        logger.info("Entered the start_data_ingestion method")
        try:
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initate_data_ingestion()
            logger.info("Got train test csv")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        logger.info("Entered the start_data_ingestion method")
        try:
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                           data_validation_config=self.data_validation_config)
            data_validation_artifact=data_validation.initiate_data_validation()
            logger.info("Performed the data validation opertion")

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e)

    def run_pipeline(self)->None:
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            logger.info("Pipeline finished successfully!")
        except Exception as e:
            raise CustomException(e)
        
        


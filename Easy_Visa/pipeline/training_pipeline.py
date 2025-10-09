import sys
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from Easy_Visa.components.data_ingestion import DataIngestion
from Easy_Visa.components.data_validation import DataValidation
from Easy_Visa.components.data_transformation import DataTransformation
from Easy_Visa.components.model_trainer import ModelTrainer

from Easy_Visa.entity.config_entity import (DataIngestionConfig,DataValidationConfig,
                                            DataTransformationConfig,ModelTrainerConfig)
from Easy_Visa.entity.artifact_entity import (DataIngestionArtifact,DataValidationArtifact,
                                              DataTransformationArtifact,ModelTrainerArtifact)


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()
        self.data_transformation_config=DataTransformationConfig()
        self.model_trainer_config=ModelTrainerConfig()

    def start_data_ingestion(self)->DataIngestionArtifact:
        logger.info("Entered the start_data_ingestion operation")
        try:
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initate_data_ingestion()
            logger.info("Got train test csv")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        logger.info("Entered the start_data_validation operation")
        try:
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                           data_validation_config=self.data_validation_config)
            data_validation_artifact=data_validation.initiate_data_validation()
            logger.info("Performed the data validation operation")

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        logger.info("Entered the start_data_transformation operation")
        try:
            data_transformation=DataTransformation(data_validation_artifact=data_validation_artifact,
                                           data_transformation_config=self.data_transformation_config)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logger.info("Performed the data transformation opertion")

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        logger.info("Entered the start_model_trainer operation")
        try:
            model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                       model_trainer_config=self.model_trainer_config)
            model_trainer_artifact=model_trainer.initiate_model_trainer()

            logger.info("Performed the model_trainer opertion")

            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e)

    def run_pipeline(self)->None:
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            logger.info("Training pipeline completed successfully!")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise CustomException(e)

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        sys.exit(1)
        
        


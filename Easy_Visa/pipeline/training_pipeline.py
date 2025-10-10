import sys
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from Easy_Visa.components.data_ingestion import DataIngestion
from Easy_Visa.components.data_validation import DataValidation
from Easy_Visa.components.data_transformation import DataTransformation
from Easy_Visa.components.model_trainer import ModelTrainer
from Easy_Visa.components.model_evaluation import ModelEvaluation
from Easy_Visa.components.model_pusher import ModelPusher

from Easy_Visa.entity.config_entity import (DataIngestionConfig,DataValidationConfig,
                                            DataTransformationConfig,ModelTrainerConfig,
                                            ModelEvaluationConfig,ModelPusherConfig)
from Easy_Visa.entity.artifact_entity import (DataIngestionArtifact,DataValidationArtifact,
                                              DataTransformationArtifact,ModelTrainerArtifact,
                                              ModelEvaluationArtifact,ModelPusherArtifact)


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        self.data_validation_config=DataValidationConfig()
        self.data_transformation_config=DataTransformationConfig()
        self.model_trainer_config=ModelTrainerConfig()
        self.model_evaluation_config=ModelEvaluationConfig()
        self.model_pusher_config=ModelPusherConfig()

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

    def start_model_evaluation(self,data_ingestion_artifact:DataIngestionArtifact,
                               model_trainer_artifact:ModelTrainerArtifact) -> ModelEvaluationArtifact:
        logger.info("Entered the start_model_evaluation operation")
        try:
            model_evaluation=ModelEvaluation(data_ingestion_artifact=data_ingestion_artifact,
                                             model_trainer_artifact=model_trainer_artifact,
                                             model_eval_config=self.model_evaluation_config)
            model_evaluation_artifact=model_evaluation.initiate_model_evaluation()

            logger.info("Performed the model_evaluation opertion")

            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e)
        
    def start_model_pusher(self,model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact:
        logger.info("Entered the start_model_pusher operation")
        try:
            model_pusher=ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                     model_pusher_config=self.model_pusher_config)
            model_pusher_artifact=model_pusher.initiate_model_pusher()

            logger.info("Performed the model_pusher opertion")

            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e)

    def run_pipeline(self)->None:
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact=self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                  model_trainer_artifact=model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                logger.info("Model not accepted")
                return None
            logger.info("Model accepted. Starting model pushing process.")
            model_pusher_artifact=self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
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
        
        


import sys,os
import json
import shutil
from datetime import datetime
from Easy_Visa.constants import MODEL_S3_FILE_NAME

from Easy_Visa.cloud_storage.aws_storage import S3Operations
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from Easy_Visa.entity.config_entity import ModelPusherConfig
from Easy_Visa.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
from Easy_Visa.utils.ml_utils.model.s3_estimator import USvisaEstimator

class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = S3Operations()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        # Define where the model will live in S3
        s3_model_path = f"{model_pusher_config.s3_model_dir}/{model_pusher_config.model_file_name}"
        self.usvisa_estimator = USvisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=s3_model_path)
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logger.info("Starting Model Pusher process...")

            # --- Prepare local folder ---
            local_dir = self.model_pusher_config.model_pusher_dir
            os.makedirs(local_dir, exist_ok=True)

            # --- Save local copies ---
            local_model_path = os.path.join(local_dir, "model.pkl")
            shutil.copy(self.model_evaluation_artifact.trained_model_path, local_model_path)

            local_metrics_path = os.path.join(local_dir, "evaluation_metrics.json")
            shutil.copy(self.model_evaluation_artifact.evaluation_metrics_path, local_metrics_path)

            metadata_path = os.path.join(local_dir, "metadata.json")
            pushed_at = datetime.now().isoformat()
            metadata = {
                "pushed_at": pushed_at,
                "s3_model_path": self.usvisa_estimator.model_path
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            # --- Load evaluation metrics to check acceptance status ---
            with open(local_metrics_path, "r") as f:
                evaluation_data = json.load(f)

            # --- Upload to S3 ---
            logger.info(f"Uploading model and metadata to S3 bucket: {self.model_pusher_config.bucket_name}")
            # Upload the model file
            self.usvisa_estimator.save_model(from_file=local_model_path, remove_local=False)
            # Upload evaluation metrics
            metrics_s3_key = f"{self.model_pusher_config.s3_model_dir}/evaluation_metrics.json"
            metadata_s3_key = f"{self.model_pusher_config.s3_model_dir}/metadata.json"
            self.s3.upload_file(local_metrics_path,
                    to_s3_key=metrics_s3_key,
                    bucket_name=self.model_pusher_config.bucket_name,
                    remove_local=False)
            # Upload metadata
            self.s3.upload_file(metadata_path,
                    to_s3_key=metadata_s3_key,
                    bucket_name=self.model_pusher_config.bucket_name,
                    remove_local=False)
            
            logger.info(f"Model pushed successfully to s3://{self.model_pusher_config.bucket_name}/{self.usvisa_estimator.model_path}")
            logger.info("Model Pusher process completed successfully.")

            # --- Return artifact ---
            return ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.usvisa_estimator.model_path,
                pushed_at=pushed_at
            )

        except Exception as e:
            raise CustomException(e)
        
if __name__ == "__main__":
        
    try:
        logger.info("Starting Model Pusher component execution")
    
        # 1. Initialize Configuration
        pusher_config = ModelPusherConfig()

        model_evaluation_artifact=ModelEvaluationArtifact(
            is_model_accepted=True,
            s3_model_path="artifacts/model_trainer/trained_model/model.pkl",
            trained_model_path="artifacts/model_trainer/trained_model/model.pkl",
            evaluation_metrics_path="artifacts/model_evaluation/evaluation_metrics.json")

        model_pusher = ModelPusher(
            model_evaluation_artifact=model_evaluation_artifact,
            model_pusher_config=pusher_config
        )
        
        pusher_artifact = model_pusher.initiate_model_pusher()
        logger.info(f"Model pushed successfully")

    except Exception as e:
        raise CustomException(e)

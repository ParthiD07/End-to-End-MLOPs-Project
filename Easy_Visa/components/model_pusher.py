import sys,os
import json
import shutil
from datetime import datetime

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
        self.usvisa_estimator = USvisaEstimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=os.path.join(model_pusher_config.s3_model_dir,
                                    model_pusher_config.version_prefix,
                                    model_pusher_config.model_file_name))
        
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
            local_dir = self.model_pusher_config.local_model_pusher_dir
            os.makedirs(local_dir, exist_ok=True)

            # --- Save local copies ---
            local_model_path = os.path.join(local_dir, "model.pkl")
            shutil.copy(self.model_evaluation_artifact.trained_model_path, local_model_path)

            local_metrics_path = os.path.join(local_dir, "evaluation_metrics.json")
            shutil.copy(self.model_evaluation_artifact.evaluation_metrics_path, local_metrics_path)

            local_best_model_reference_path = os.path.join(local_dir, "best_model_reference.json")
            shutil.copy(self.model_evaluation_artifact.best_model_reference_path, local_best_model_reference_path)

            metadata_path = os.path.join(local_dir, "metadata.json")
            model_version = self.model_pusher_config.version_prefix
            pushed_at = datetime.now().isoformat()
            metadata = {
                "model_version": model_version,
                "pushed_at": pushed_at,
                "s3_model_path": self.usvisa_estimator.model_path
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)

            # --- Upload to S3 ---
            self.usvisa_estimator.save_model(from_file=local_model_path)
            self.s3.upload_file(local_metrics_path,
                                bucket_name=self.model_pusher_config.bucket_name,
                                key=os.path.join(self.model_pusher_config.s3_model_dir, model_version, "evaluation_metrics.json"))
            self.s3.upload_file(local_best_model_reference_path,
                                bucket_name=self.model_pusher_config.bucket_name,
                                key=os.path.join(self.model_pusher_config.s3_model_dir, model_version, "best_model_reference.json"))
            self.s3.upload_file(metadata_path,
                                bucket_name=self.model_pusher_config.bucket_name,
                                key=os.path.join(self.model_pusher_config.s3_model_dir, model_version, "metadata.json"))

            logger.info("Model Pusher process completed successfully.")

            # --- Return artifact ---
            return ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.usvisa_estimator.model_path,
                local_model_path=local_model_path,
                local_metrics_path=local_metrics_path,
                local_best_model_reference_path=local_best_model_reference_path,
                local_metadata_path=metadata_path,
                model_version=model_version,
                pushed_at=pushed_at
            )

        except Exception as e:
            raise CustomException(e)

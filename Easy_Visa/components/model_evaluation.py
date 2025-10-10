import sys,os
import json
import pandas as pd
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from Easy_Visa.entity.config_entity import ModelEvaluationConfig
from Easy_Visa.components.model_pusher import ModelPusher
from Easy_Visa.entity.config_entity import ModelEvaluationConfig, ModelPusherConfig
from Easy_Visa.entity.artifact_entity import DataIngestionArtifact,ModelTrainerArtifact,ModelEvaluationArtifact

from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger

from Easy_Visa.constants import TARGET_COLUMN

from Easy_Visa.utils.ml_utils.model.estimator import VisaModel
from Easy_Visa.utils.ml_utils.model.s3_estimator import USvisaEstimator


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    test_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float
    failing_cases_path: Optional[str] = None

class ModelEvaluation:

    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact, model_eval_config: ModelEvaluationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_eval_config = model_eval_config

        except Exception as e:
            raise CustomException(e)

    def get_best_model(self) -> Optional[USvisaEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            usvisa_estimator = USvisaEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if usvisa_estimator.is_model_present():
                return usvisa_estimator
            return None
        except Exception as e:
            raise  CustomException(e)

    @staticmethod
    def transform_target_column(target_series:pd.Series)->pd.Series:
        """Convert target labels to numeric"""
        try:
            mapping = {'Certified': 1, 'Denied': 0}
            return target_series.map(mapping)
        except Exception as e:
            raise CustomException(e)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            current_year = datetime.now().year
            test_df['company_age'] = current_year-test_df['yr_of_estab']

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            
            y = ModelEvaluation.transform_target_column(y)

            with open(self.model_trainer_artifact.train_metric_file_path, "r") as f:
                train_metrics = json.load(f)

            with open(self.model_trainer_artifact.test_metric_file_path, "r") as f:
                test_metrics = json.load(f)

            trained_model_f1_score = train_metrics["f1_score"]
            test_model_f1_score = test_metrics["f1_score"]

            best_model_f1_score=None
            failing_cases_path = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)

                 # --- Generate failing cases ---
                failing_cases = test_df[y != y_hat_best_model]
                if not failing_cases.empty:
                    eval_dir = os.path.join("artifacts", "model_evaluation")
                    os.makedirs(eval_dir, exist_ok=True)
                    failing_cases_path = os.path.join(eval_dir, "failing_cases.csv")
                    failing_cases.to_csv(failing_cases_path, index=False)

                    # --- Plot confusion matrix ---
                    cm = confusion_matrix(y, y_hat_best_model)
                    plt.figure(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix - Production Model")
                    cm_path = os.path.join(eval_dir, "confusion_matrix.png")
                    plt.savefig(cm_path)
                    plt.close()
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           test_model_f1_score=test_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logger.info(f"Result: {result}")
            return result

        except Exception as e:
            raise CustomException(e)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            eval_response = self.evaluate_model()
            eval_dir = os.path.join("artifacts", "model_evaluation")
            os.makedirs(eval_dir, exist_ok=True)

            # --- Save evaluation metrics ---
            metrics_path = os.path.join(eval_dir, "evaluation_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({
                    "trained_model_f1_score": float(eval_response.trained_model_f1_score),
                    "test_model_f1_score": float(eval_response.test_model_f1_score),
                    "best_model_f1_score": float(eval_response.best_model_f1_score) if eval_response.best_model_f1_score is not None else None,
                    "is_model_accepted": bool(eval_response.is_model_accepted),
                    "difference": float(eval_response.difference),
                    "failing_cases_path": eval_response.failing_cases_path
                }, f, indent=4)

            # --- Save best model reference ---
            best_model_ref_path = os.path.join(eval_dir, "best_model_reference.json")
            with open(best_model_ref_path, "w") as f:
                json.dump({"s3_model_path": self.model_eval_config.s3_model_key_path}, f, indent=4)

            # --- Add confusion matrix path if generated ---
            confusion_matrix_path = None
            if os.path.exists(os.path.join(eval_dir, "confusion_matrix.png")):
                confusion_matrix_path = os.path.join(eval_dir, "confusion_matrix.png")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=eval_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=eval_response.difference,
                failing_cases_path=eval_response.failing_cases_path,
                evaluation_metrics_path=metrics_path,          
                best_model_reference_path=best_model_ref_path,
                confusion_matrix_path=confusion_matrix_path
            )

            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e)
        
if __name__=="__main__":
    try:
        logger.info("Starting Model Evaluation component execution")
        eval_config=ModelEvaluationConfig()
        pusher_config = ModelPusherConfig()
        data_ingestion_artifact=DataIngestionArtifact(train_file_path="artifacts/data_ingestion/ingested/train.csv",
                                                      test_file_path="artifacts/data_ingestion/ingested/test.csv")

        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path="artifacts/model_trainer/trained_model/model.pkl",
                                                    train_metric_file_path="artifacts/model_trainer/trained_model/metrics/train_metrics.json",
                                                    test_metric_file_path="artifacts/model_trainer/trained_model/metrics/test_metrics.json")

        model_evaluator=ModelEvaluation(data_ingestion_artifact=data_ingestion_artifact,
                                         model_trainer_artifact=model_trainer_artifact,
                                         model_eval_config=eval_config)

        eval_artifact = model_evaluator.initiate_model_evaluation()
        logger.info("Model Evaluation finished successfully")
        # --- Push Model to S3 if accepted ---
        if eval_artifact.is_model_accepted:
            logger.info("Model accepted. Starting Model Pusher...")
            model_pusher = ModelPusher(
                model_evaluation_artifact=eval_artifact,
                model_pusher_config=pusher_config
            )
            
            pusher_artifact = model_pusher.initiate_model_pusher()
            logger.info(f"Model pushed successfully: {pusher_artifact}")
        else:
            logger.warning("Model not accepted. Skipping Model Pusher stage.")
            
    except Exception as e:
        logger.error(f"Pipeline failed! Error: {e}")
        raise CustomException(e)
        
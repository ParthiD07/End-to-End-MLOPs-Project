from datetime import datetime
import os
from Easy_Visa.constants import *

class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_dir: str=os.path.join(ARTIFACTS_DIR,DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str=os.path.join(self.data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
        self.training_file_path: str=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)  
        self.testing_file_path: str=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
        self.train_test_ratio: float=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.database_name: str=DATA_INGESTION_DATABASE_NAME
        self.collection_name: str=DATA_INGESTION_COLLECTION_NAME

class DataValidationConfig:
    def __init__(self):
        self.data_validation_dir: str=os.path.join(ARTIFACTS_DIR,DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str=os.path.join(self.data_validation_dir,DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str=os.path.join(self.data_validation_dir,DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str=os.path.join(self.valid_data_dir,TRAIN_FILE_NAME)
        self.valid_test_file_path: str=os.path.join(self.valid_data_dir,TEST_FILE_NAME)
        self.invalid_train_file_path: str=os.path.join(self.invalid_data_dir,TRAIN_FILE_NAME)
        self.invalid_test_file_path: str=os.path.join(self.invalid_data_dir,TEST_FILE_NAME)
        self.drift_report_file_path: str=os.path.join(self.data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,
                                                      DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

class DataTransformationConfig:
    def __init__(self):
        self.data_transformation_dir: str= os.path.join(ARTIFACTS_DIR,DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str= os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                            TRAIN_FILE_NAME.replace("csv","npy"))
        self.transformed_test_file_path: str= os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                           TEST_FILE_NAME.replace("csv","npy"))
        self.transformed_object_file_path: str=os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                            PREPROCESSING_OBJECT_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self):
        self.model_trainer_dir: str=os.path.join(ARTIFACTS_DIR,MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path: str=os.path.join(self.model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_FILE_NAME)
        self.expected_accuracy: float=MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold: float=MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD


class ModelEvaluationConfig:
    def __init__(self):
        self.model_evaluation_dir: str = os.path.join(ARTIFACTS_DIR,MODEL_EVALUATION_DIR_NAME)       
        self.changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        self.bucket_name: str = MODEL_S3_BUCKET_NAME
        self.s3_model_key_path: str = MODEL_FILE_NAME
        

class ModelPusherConfig:
    def __init__(self):
        self.bucket_name: str = MODEL_S3_BUCKET_NAME
        self.model_pusher_dir: str = os.path.join(ARTIFACTS_DIR, "model_pusher")
        self.model_file_name: str = MODEL_FILE_NAME
        self.s3_model_dir: str = "models/us_visa_classifier"
        self.version_prefix: str = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"

class USVisaPredictionConfig:
    def __init__(self):
        self.model_file_path: str = MODEL_FILE_NAME
        self.model_bucket_name: str = MODEL_S3_BUCKET_NAME
        self.local_prediction_dir: str = os.path.join("artifacts", "prediction")
        os.makedirs(self.local_prediction_dir, exist_ok=True)
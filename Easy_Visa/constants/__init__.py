
from datetime import date
import os
"""
defining common constant variable for training pipeline
"""


PIPELINE_NAME: str="EasyVisa"
ARTIFACTS_DIR: str="artifacts"
FILE_NAME: str="EasyVisa.csv"

TRAIN_FILE_NAME: str="train.csv"
TEST_FILE_NAME: str="test.csv"

TARGET_COLUMN: str="case_status"

SCHEMA_FILE_PATH=os.path.join("config","schema.yaml")


"""
Data ingestion related to constant
"""
DATA_INGESTION_DATABASE_NAME: str="Easy_Visa"
DATA_INGESTION_COLLECTION_NAME: str="visa_data"
DATA_INGESTION_DIR_NAME: str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str="feature_store"
DATA_INGESTION_INGESTED_DIR: str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float= 0.2

"""
Data validation related to constant
"""

DATA_VALIDATION_DIR_NAME: str="data_validation"
DATA_VALIDATION_VALID_DIR: str="validated"
DATA_VALIDATION_INVALID_DIR: str="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str="report.yaml"

"""
Data transformation related to constant
"""

DATA_TRANSFORMATION_DIR_NAME: str="data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str="transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str="transformed_object"
PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"

"""
Model trainer related to constant
"""

MODEL_TRAINER_DIR_NAME: str="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str="trained_model"
MODEL_FILE_NAME: str="model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float= 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float=0.05

"""
Model evaluation related to constant
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation" 
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_S3_BUCKET_NAME: str = "us-visa-model-8321-9489-5652"
EVALUATION_METRICS_FILE_NAME: str = "evaluation_metrics.json" 


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

MODEL_FILE_NAME: str="model.pkl"

TARGET_COLUMN: str="case_status"
CURRENT_YEAR: str=date.today().year
PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"
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
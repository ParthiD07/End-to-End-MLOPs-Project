
"""
defining common constant variable for training pipeline
"""

TARGET: str="case_status"
PIPELINE_NAME: str="EasyVisa"
ARTIFACTS_DIR: str="artifacts"
FILE_NAME: str="EasyVisa.csv"

TRAIN_FILE_NAME: str="train.csv"
TEST_FILE_NAME: str="test.csv"

"""
Data ingestion related to constant
"""
DATA_INGESTION_DATABASE_NAME: str="Easy_Visa"
DATA_INGESTION_COLLECTION_NAME: str="visa_data"
DATA_INGESTION_DIR_NAME: str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str="feature_store"
DATA_INGESTION_INGESTED_DIR: str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float= 0.2
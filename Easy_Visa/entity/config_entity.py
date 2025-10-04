from datetime import datetime
import os
from Easy_Visa.constants import *

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp=timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.timestamp=timestamp
        self.pipeline_name: str=PIPELINE_NAME
        self.artifact_name: str=ARTIFACTS_DIR
        self.artifact_dir: str=os.path.join(self.artifact_name,timestamp)


class DataIngestionConfig:
    def __init__(self,train_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str=os.path.join(train_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str=os.path.join(self.data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
        self.training_file_path: str=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)  
        self.testing_file_path: str=os.path.join(self.data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
        self.train_test_ratio: float=DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.database_name: str=DATA_INGESTION_DATABASE_NAME
        self.collection_name: str=DATA_INGESTION_COLLECTION_NAME

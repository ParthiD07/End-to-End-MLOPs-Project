import os
import sys
import pandas as pd
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split

from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger

# configuration of the data ingestion config
from Easy_Visa.entity.config_entity import DataIngestionConfig
from Easy_Visa.entity.artifact_entity import DataIngestionArtifact

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CustomException(e)
        
    def export_collection_as_dataframe(self):
        """"""
        database_name=self.data_ingestion_config.database_name
        collection_name=self.data_ingestion_config.collection_name
        self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
        collection=self.mongo_client[database_name][collection_name]

        df=pd.DataFrame(list(collection.find()))
        if "_id" in df.columns.to_list():
            df=df.drop(columns=["_id"],axis=1)

        df.replace("na",np.nan,inplace=True)
        return df

    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path=os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise CustomException(e)
        
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_ratio)
            logger.info("Performed train test split on the dataframe")
            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logger.info("Exporting train and test file to ingested path")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logger.info("train and test file exported to path")

        except Exception as e:
            raise CustomException(e)

    def initate_data_ingestion(self):
        try:
            os.makedirs(self.data_ingestion_config.data_ingestion_dir, exist_ok=True)
            logger.info(f"Created DVC tracking root: {self.data_ingestion_config.data_ingestion_dir}")

            # Now proceed with data logic
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifact=DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                          test_file_path=self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e)
import os

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")

import certifi
ca=certifi.where() # contains trusted certification authority

import pandas as pd
import pymongo
from Easy_Visa.logging.logger import logger
from Easy_Visa.exception.exception import CustomException

class EasyVisaDataExtract:
    def __init__(self, database: str, collection: str):
        """
        Initialize MongoDB connection and set up database/collection.

        Args:
            database (str): Name of the MongoDB database.
            collection (str): Name of the MongoDB collection.
        """
        try:
            logger.info(f"Initializing MongoDB client for {database}.{collection}")
            self.client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.db = self.client[database]
            self.collection = self.db[collection]
        except Exception as e:
            raise CustomException(e)
        
    def csv_to_json_converter(self,file_path):
        """Convert CSV file into list of JSON-like dict records"""
        logger.info(f"Reading CSV file: {file_path}")

        try:
            data=pd.read_csv(file_path)
            records=data.to_dict(orient="records")
            logger.info(f"Successfully converted {len(records)} rows to JSON records")
            return records
        except Exception as e:
            raise CustomException(e)
        
    def insert_data_to_mongodb(self,records):
        """Insert list of records into MongoDB collection.
        If collection already has data, skip insertion."""
        try:
            existing_count=self.collection.count_documents({})
            if existing_count > 0:
                logger.info(f"Collection already has {existing_count} records. Skipping insertion.")
                return 0
            else: 
                result = self.collection.insert_many(records)
                logger.info(f"Inserted {len(result.inserted_ids)} records into MongoDB.")
                return len(result.inserted_ids)
        except Exception as e:
            raise CustomException(e)

if __name__ == "__main__":
    extractor = EasyVisaDataExtract(database="Easy_Visa", collection="visa_data")
    records = extractor.csv_to_json_converter("CSV_Data\EasyVisa.csv")
    extractor.insert_data_to_mongodb(records)


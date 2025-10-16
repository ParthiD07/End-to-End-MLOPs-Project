import boto3
import os
import pickle
from io import StringIO
from typing import Any, Optional, Union
import sys

from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv
from Easy_Visa.configuration.aws_connection import S3Client


class S3Operations:
    def __init__(self, profile_name: str = "default", region_name: str = "us-east-1"):
        try:
            s3_client = S3Client()
            self.s3_resource = s3_client.resource
            self.s3_client = s3_client.client
        except Exception as e:
            raise CustomException(e)

    # --- Helper Methods ---
    
    @staticmethod
    def normalize_s3_key(key: str) -> str:
        """
        Normalize S3 key to use forward slashes and remove leading slashes.
        S3 keys should always use forward slashes, never backslashes.
        """
        # Replace backslashes with forward slashes
        normalized = key.replace("\\", "/")
        # Remove leading slash if present (S3 keys shouldn't start with /)
        normalized = normalized.lstrip("/")
        # Remove any double slashes
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        return normalized

    # --- Bucket Helpers ---

    def get_bucket(self, bucket_name: str):
        """Return a boto3 Bucket object."""
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise CustomException(e)

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        """Check if any object exists at the given S3 key prefix."""
        try:
            # Normalize the key before checking
            s3_key = self.normalize_s3_key(s3_key)
            
            logger.info(f"Checking if S3 key exists: s3://{bucket_name}/{s3_key}")
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=s3_key,
                MaxKeys=1
            )
            exists = "Contents" in response
            logger.info(f"S3 key exists: {exists}")
            return exists
        except Exception as e:
            raise CustomException(e)

    def get_file_object(self, s3_key: str, bucket_name: str) -> Optional[Any]:
        """Get S3 Object; return None if not found."""
        try:
            # Normalize the key
            s3_key = self.normalize_s3_key(s3_key)
            
            obj = self.s3_resource.Object(bucket_name, s3_key)
            obj.load()  # check existence
            logger.info(f"Successfully retrieved S3 object: s3://{bucket_name}/{s3_key}")
            return obj
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning(f"File not found: s3://{bucket_name}/{s3_key}")
                return None
            raise CustomException(e)
        except Exception as e:
            raise CustomException(e)

    def read_object(self, file_object: Any, decode: bool = True, make_readable: bool = False) -> Union[StringIO, bytes, str]:
        """Read S3 object content."""
        try:
            response = file_object.get()
            data = response["Body"].read()
            if decode:
                content = data.decode("utf-8")
                return StringIO(content) if make_readable else content
            return data
        except Exception as e:
            raise CustomException(e)

    # --- Model Operations ---

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """
        Load pickled model from S3.
        
        Args:
            model_name: Full S3 key path or just filename
            bucket_name: S3 bucket name
            model_dir: Optional directory prefix (deprecated - include in model_name)
        """
        try:
            # If model_dir is provided, join it with model_name
            if model_dir:
                s3_key = f"{model_dir}/{model_name}"
            else:
                s3_key = model_name
            
            # Normalize to ensure forward slashes
            s3_key = self.normalize_s3_key(s3_key)
            
            logger.info(f"Attempting to load model from s3://{bucket_name}/{s3_key}")
            
            file_object = self.get_file_object(s3_key, bucket_name)
            if file_object is None:
                raise FileNotFoundError(f"Model not found at s3://{bucket_name}/{s3_key}")
            
            model_bytes = self.read_object(file_object, decode=False)
            model = pickle.loads(model_bytes)
            
            logger.info(f"Successfully loaded model from S3")
            return model
        except Exception as e:
            raise CustomException(e)

    def upload_file(self, from_local_path: str, to_s3_key: str, bucket_name: str, remove_local: bool = True) -> None:
        """Upload local file to S3 and optionally remove local copy."""
        try:
            # Normalize S3 key to use forward slashes
            to_s3_key = self.normalize_s3_key(to_s3_key)
            
            logger.info(f"Uploading {from_local_path} to s3://{bucket_name}/{to_s3_key}")
            
            self.s3_client.upload_file(from_local_path, bucket_name, to_s3_key)
            
            logger.info(f"Successfully uploaded to s3://{bucket_name}/{to_s3_key}")
            
            if remove_local:
                os.remove(from_local_path)
                logger.info(f"Deleted local file: {from_local_path}")
        except Exception as e:
            raise CustomException(e)

    def upload_df_as_csv(self, data_frame: DataFrame, bucket_filename: str, bucket_name: str) -> None:
        """Upload DataFrame to S3 as CSV without writing local file."""
        try:
            # Normalize S3 key
            bucket_filename = self.normalize_s3_key(bucket_filename)
            
            logger.info(f"Uploading DataFrame to s3://{bucket_name}/{bucket_filename}")
            csv_buffer = StringIO()
            data_frame.to_csv(csv_buffer, index=False)
            self.s3_client.put_object(Bucket=bucket_name, Key=bucket_filename, Body=csv_buffer.getvalue())
            
            logger.info(f"Successfully uploaded DataFrame to S3")
        except Exception as e:
            raise CustomException(e)

    def read_csv_to_df(self, s3_key: str, bucket_name: str) -> DataFrame:
        """Read CSV from S3 into DataFrame."""
        try:
            # Normalize key
            s3_key = self.normalize_s3_key(s3_key)
            
            file_object = self.get_file_object(s3_key, bucket_name)
            if file_object is None:
                raise FileNotFoundError(f"CSV not found at s3://{bucket_name}/{s3_key}")
            content = self.read_object(file_object, make_readable=True)
            return read_csv(content, na_values="na")
        except Exception as e:
            raise CustomException(e)

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """Create a 'folder' in S3 (zero-byte object ending with '/')."""
        try:
            # Normalize and ensure trailing slash
            folder_obj = self.normalize_s3_key(folder_name)
            folder_obj = folder_obj.rstrip("/") + "/"
            
            try:
                self.s3_resource.Object(bucket_name, folder_obj).load()
                logger.info(f"Folder already exists: s3://{bucket_name}/{folder_obj}")
                return
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    raise
            
            self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
            logger.info(f"Created folder: s3://{bucket_name}/{folder_obj}")
        except Exception as e:
            raise CustomException(e)

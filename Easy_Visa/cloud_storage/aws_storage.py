import boto3
import os
import pickle
from io import StringIO
from typing import Any, Optional, Union

from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv


class S3Operations:
    def __init__(self, profile_name: str = "default", region_name: str = "us-east-1"):
        try:
            session = boto3.Session(profile_name=profile_name, region_name=region_name)
            self.s3_resource = session.resource("s3")
            self.s3_client = session.client("s3")
            self.Bucket = self.s3_resource.Bucket
        except Exception as e:
            raise CustomException(e)

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
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=s3_key,
                MaxKeys=1
            )
            return "Contents" in response
        except Exception as e:
            raise CustomException(e)

    def get_file_object(self, s3_key: str, bucket_name: str) -> Optional[Any]:
        """Get S3 Object; return None if not found."""
        try:
            obj = self.s3_resource.Object(bucket_name, s3_key)
            obj.load()  # check existence
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
        """Load pickled model from S3."""
        try:
            s3_key = os.path.join(model_dir or "", model_name).replace("\\", "/")
            file_object = self.get_file_object(s3_key, bucket_name)
            if file_object is None:
                raise FileNotFoundError(f"Model not found at s3://{bucket_name}/{s3_key}")
            model_bytes = self.read_object(file_object, decode=False)
            return pickle.loads(model_bytes)
        except Exception as e:
            raise CustomException(e)

    def upload_file(self, from_local_path: str, to_s3_key: str, bucket_name: str, remove_local: bool = True) -> None:
        """Upload local file to S3 and optionally remove local copy."""
        try:
            logger.info(f"Uploading {from_local_path} to s3://{bucket_name}/{to_s3_key}")
            self.s3_client.upload_file(from_local_path, bucket_name, to_s3_key)
            if remove_local:
                os.remove(from_local_path)
                logger.info(f"Deleted local file: {from_local_path}")
        except Exception as e:
            raise CustomException(e)

    def upload_df_as_csv(self, data_frame: DataFrame, bucket_filename: str, bucket_name: str) -> None:
        """Upload DataFrame to S3 as CSV without writing local file."""
        try:
            logger.info(f"Uploading DataFrame to s3://{bucket_name}/{bucket_filename}")
            csv_buffer = StringIO()
            data_frame.to_csv(csv_buffer, index=False)
            self.s3_client.put_object(Bucket=bucket_name, Key=bucket_filename, Body=csv_buffer.getvalue())
        except Exception as e:
            raise CustomException(e)

    def read_csv_to_df(self, s3_key: str, bucket_name: str) -> DataFrame:
        """Read CSV from S3 into DataFrame."""
        try:
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
            folder_obj = folder_name.rstrip("/") + "/"
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

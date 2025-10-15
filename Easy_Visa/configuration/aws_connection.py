import boto3
import os
from dotenv import load_dotenv

class S3Client:

    s3_client= None
    s3_resource= None

    def __init__(self,region_name=os.getenv("us-east-1")):
        """
        Gets get aws credentials from env_variable and creates an connection with s3 bucket
        """
        if S3Client.s3_client==None or S3Client.s3_resource==None:
            
            # --- STEP 1: LOAD ENVIRONMENT VARIABLES FROM .env FILE ---
            load_dotenv()

            # --- Load Credentials & Validate ---
            access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

            if not access_key_id:
                raise Exception(f"Environment variable: {access_key_id} is not set.")
            if not secret_access_key:
                raise Exception(f"Environment variable: {secret_access_key} is not set.")

            # --- Create Connections ---
            S3Client.s3_resource = boto3.resource(
                                    's3',
                                    aws_access_key_id=access_key_id,
                                    aws_secret_access_key=secret_access_key,
                                    region_name=region_name
                                )
            S3Client.s3_client = boto3.client(
                                    's3',
                                    aws_access_key_id=access_key_id,
                                    aws_secret_access_key=secret_access_key,
                                    region_name=region_name
                                )
        # Assign the class-level connections to instance attributes
        self.resource = S3Client.s3_resource
        self.client = S3Client.s3_client


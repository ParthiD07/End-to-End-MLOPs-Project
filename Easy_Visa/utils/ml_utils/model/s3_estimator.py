from Easy_Visa.cloud_storage.aws_storage import S3Operations
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.utils.ml_utils.model.estimator import VisaModel
from Easy_Visa.logging.logger import logger
import sys
from pandas import DataFrame


class USvisaEstimator:
    """
    Handles saving, retrieving, and using the EasyVisa model stored in S3.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        Initialize EasyVisaEstimator for S3 model operations.

        Args:
            bucket_name (str): Name of the S3 bucket.
            model_path (str): S3 key/path to the model file (e.g., 'us_visa_files/model.pkl').
        """
        self.s3 = S3Operations()
        # Normalize model path immediately
        self.model_path = S3Operations.normalize_s3_key(model_path)
        self.bucket_name = bucket_name
        self.loaded_model: VisaModel = None
        
        logger.info(f"Initialized USvisaEstimator with bucket={bucket_name}, path={self.model_path}")

    def is_model_present(self) -> bool:
        """Check if model exists in S3."""
        try:
            exists = self.s3.s3_key_path_available(
                bucket_name=self.bucket_name, 
                s3_key=self.model_path
            )
            logger.info(f"Model exists check for s3://{self.bucket_name}/{self.model_path}: {exists}")
            return exists
        except Exception as e:
            raise CustomException(e)

    def load_model(self) -> VisaModel:
        """
        Load the model from S3.

        Returns:
            VisaModel: The loaded model object.
        """
        try:
            logger.info(f"Loading model from s3://{self.bucket_name}/{self.model_path}")
            
            # Check if model exists first
            if not self.is_model_present():
                raise FileNotFoundError(
                    f"Model not found at s3://{self.bucket_name}/{self.model_path}. "
                    "Please run the training pipeline first to upload the model."
                )
            
            # Load model (pass full path as model_name, no model_dir)
            self.loaded_model = self.s3.load_model(
                model_name=self.model_path,
                bucket_name=self.bucket_name,
                model_dir=None  # Don't use model_dir since we have full path
            )
            
            logger.info("Model loaded successfully from S3")
            return self.loaded_model
        except Exception as e:
            raise CustomException(e)

    def save_model(self, from_file: str, remove_local: bool = False) -> None:
        """
        Upload a trained model to S3.

        Args:
            from_file (str): Local file path of the trained model.
            remove_local (bool): If True, delete the local file after uploading.
        """
        try:
            logger.info(f"Saving model from {from_file} to s3://{self.bucket_name}/{self.model_path}")
            
            self.s3.upload_file(
                from_file,
                to_s3_key=self.model_path,
                bucket_name=self.bucket_name,
                remove_local=remove_local
            )
            
            logger.info("Model saved successfully to S3")
        except Exception as e:
            raise CustomException(e)

    def predict(self, dataframe: DataFrame):
        """
        Perform predictions using the loaded model.

        Args:
            dataframe (DataFrame): Input data for prediction.

        Returns:
            ndarray or list: Model predictions.
        """
        try:
            if self.loaded_model is None:
                logger.info("Model not loaded in memory, loading from S3...")
                self.loaded_model = self.load_model()
            
            logger.info(f"Making predictions on {len(dataframe)} samples")
            predictions = self.loaded_model.predict(dataframe=dataframe)
            logger.info("Predictions completed successfully")
            
            return predictions
        except Exception as e:
            raise CustomException(e)

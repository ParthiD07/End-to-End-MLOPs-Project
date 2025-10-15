from Easy_Visa.cloud_storage.aws_storage import S3Operations
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.utils.ml_utils.model.estimator import VisaModel
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
            model_path (str): S3 key/path to the model file (e.g., 'models/easyvisa_model.pkl').
        """
        
        self.s3 = S3Operations()
        self.model_path = model_path
        self.bucket_name = bucket_name
        self.loaded_model:VisaModel = None


    def is_model_present(self):
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=self.model_path)
        except Exception as e:
            raise CustomException(e)

    def load_model(self)->VisaModel:
        """
        Load the model from S3.

        Returns:
            VisaModel: The loaded model object.
        """
        try:
            self.loaded_model = self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
            return self.loaded_model
        except Exception as e:
            raise CustomException(e)

    def save_model(self,from_file,remove_local:bool=False)->None:
        """
        Upload a trained model to S3.

        Args:
            from_file (str): Local file path of the trained model.
            remove (bool): If True, delete the local file after uploading.
        """
        try:
            self.s3.upload_file(from_file,
                                to_s3_key=self.model_path,
                                bucket_name=self.bucket_name,
                                remove_local=remove_local)
        except Exception as e:
            raise CustomException(e)


    def predict(self,dataframe:DataFrame):
        """
        Perform predictions using the loaded model.

        Args:
            dataframe (DataFrame): Input data for prediction.

        Returns:
            ndarray or list: Model predictions.
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise CustomException(e)
import os
import sys

import numpy as np
import pandas as pd
from pandas import DataFrame

from Easy_Visa.entity.config_entity import USVisaPredictionConfig
from Easy_Visa.utils.ml_utils.model.s3_estimator import USvisaEstimator

from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from pandas import DataFrame


class USvisaData:
    """
    A helper class to structure and transform input data 
    into a DataFrame format suitable for prediction.
    """
    def __init__(self,
                continent: str,
                education_of_employee: str,
                has_job_experience: str,
                requires_job_training: str,
                no_of_employees: int,
                region_of_employment: str,
                prevailing_wage: float,
                unit_of_wage: str,
                full_time_position: str,
                company_age: int):

        try:
            # Validate categorical inputs
            required_str_fields = [continent, education_of_employee, has_job_experience,
                                   requires_job_training, region_of_employment,
                                   unit_of_wage, full_time_position]

            if not all(required_str_fields):
                raise ValueError("All categorical inputs must be provided and non-empty.")

            # Validate numeric inputs
            if not all(isinstance(x, (int, float)) for x in [no_of_employees, prevailing_wage, company_age]):
                raise TypeError("Numeric inputs (employees, wage, company_age) must be int or float.")
            
            self.continent = continent
            self.education_of_employee = education_of_employee
            self.has_job_experience = has_job_experience
            self.requires_job_training = requires_job_training
            self.no_of_employees = no_of_employees
            self.region_of_employment = region_of_employment
            self.prevailing_wage = prevailing_wage
            self.unit_of_wage = unit_of_wage
            self.full_time_position = full_time_position
            self.company_age = company_age

            logger.info("Initialized USvisaData instance successfully.")

        except Exception as e:
            raise CustomException(e)
        
    def get_usvisa_data_as_dict(self):
        """
        Converts the user input into a dictionary format
        compatible with the model’s input schema.
        """
        logger.info("Entered get_usvisa_data_as_dict method as USvisaData class")

        try:
            input_data = {
                "continent": [self.continent],
                "education_of_employee": [self.education_of_employee],
                "has_job_experience": [self.has_job_experience],
                "requires_job_training": [self.requires_job_training],
                "no_of_employees": [self.no_of_employees],
                "region_of_employment": [self.region_of_employment],
                "prevailing_wage": [self.prevailing_wage],
                "unit_of_wage": [self.unit_of_wage],
                "full_time_position": [self.full_time_position],
                "company_age": [self.company_age],
            }

            logger.info("Created input data dictionary successfully.")

            return input_data

        except Exception as e:
            raise CustomException(e)

    def get_usvisa_input_data_frame(self)-> DataFrame:
        """
        Converts the user input dictionary into a Pandas DataFrame
        for prediction.
        """
        try:
            logger.info("Entered get_usvisa_input_data_frame method of USvisaData class.")
            
            df = pd.DataFrame(self.get_usvisa_data_as_dict())
            logger.info(f"Created DataFrame for prediction with shape: {df.shape}")
            return df
        
        except Exception as e:
            raise CustomException(e)

class USvisaClassifier:
    def __init__(self, config: USVisaPredictionConfig=USVisaPredictionConfig()):
        """
        Initialize the prediction pipeline with configuration.
        """
        try:
            self.config = config
            logger.info("USvisaClassifier initialized successfully.")
        except Exception as e:
            raise CustomException(e)


    def predict(self, dataframe: pd.DataFrame) -> np.ndarray:
        """
        Perform prediction using the trained model from S3.
        
        Parameters:
            dataframe (pd.DataFrame): Preprocessed input data
        
        Returns:
            np.ndarray: Model predictions
        """
        try:
            if dataframe.empty:
                raise ValueError("Input dataframe is empty. Cannot perform prediction.")
            
            logger.info("Entered predict method of USvisaClassifier class")

            # Load model safely from S3
            model = USvisaEstimator(
                bucket_name=self.config.model_bucket_name,
                model_path=self.config.model_file_path)
            
            logger.info("Loaded model from S3 successfully. Starting prediction...")
            prediction =  model.predict(dataframe)
            logger.info(f"Prediction completed successfully.")
            
            # Ensure 1D array
            if prediction.ndim > 1 and prediction.shape[1] == 1:
                prediction = prediction.ravel()

            return prediction
        
        except Exception as e:
            raise CustomException(e)
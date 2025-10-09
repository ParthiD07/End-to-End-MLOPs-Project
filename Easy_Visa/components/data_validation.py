import os,sys
import json
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pandas import DataFrame

from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from Easy_Visa.entity.config_entity import DataValidationConfig
from Easy_Visa.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from Easy_Visa.utils.main_utils import read_yaml,save_yaml
from Easy_Visa.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config=read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e)

    @staticmethod
    def read_data(file_path)->DataFrame:
        """A static method is a method inside a class that does not depend on the class instance (self).
        It is used when the method performs a task that is related to the class but doesn’t need object-specific data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e)
        
    def validate_number_of_columns(self,dataframe:DataFrame)->bool:

        try:
            status=len(dataframe.columns) == len(self._schema_config["columns"])
            logger.info(f"Is required columns present: [{status}]")
            return status
        except Exception as e:
            raise CustomException(e)

    def is_columns_exist(self,dataframe:DataFrame)->bool:

        try:
            dataframe_columns = dataframe.columns
            missing_numerical_columns=[]
            missing_categorical_columns=[]

            # check numerical columns
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            # check categorical columns
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_numerical_columns)>0 or len(missing_categorical_columns)>0:
                logger.error(f"Missing numerical columns: {missing_numerical_columns}")
                logger.error(f"Missing categorical columns: {missing_categorical_columns}")
                return False
            
            logger.info("All required columns exist in dataframe.")
            return True

        except Exception as e:
            raise CustomException(e)
        
    def detect_dataset_drift(self,reference_df:DataFrame, current_df:DataFrame)->bool:

        """
        Detects data drift between reference (train) and current (test) datasets
        using Evidently's DataDriftPreset.

        Saves both YAML and HTML reports for interpretability.

        Returns:
        bool: True if drift is detected, False otherwise.
        """
        try:
            # Create Evidently report
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)

            # drift report directory
            drift_report_dir = os.path.dirname(self.data_validation_config.drift_report_file_path)
            os.makedirs(drift_report_dir, exist_ok=True)

            
            # Convert numpy and other objects to native Python types
            drift_json = report.as_dict()
            drift_json_serializable = json.loads(json.dumps(drift_json))

            # Save YAML report
            save_yaml(path_to_yaml=self.data_validation_config.drift_report_file_path, data=drift_json_serializable)
            logger.info(f"YAML Drift report saved at: {self.data_validation_config.drift_report_file_path}")

            # Save HTML report in same folder with same name but .html extension
            drift_html_path = os.path.splitext(self.data_validation_config.drift_report_file_path)[0] + ".html"
            report.save_html(drift_html_path)
            logger.info(f"HTML Drift report saved at: {drift_html_path}")


            # Extract drift metrics
            drift_summary = drift_json["metrics"][0]["result"]["dataset_drift"]
            n_drifted = drift_json["metrics"][0]["result"]["number_of_drifted_columns"]
            total_cols = drift_json["metrics"][0]["result"]["number_of_columns"]


            logger.info(f"Detected drift in {n_drifted}/{total_cols} features ({(n_drifted/total_cols)*100:.2f}%).")
            return drift_summary

        except Exception as e:
            raise CustomException(e)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            validation_error_msg=""
            logger.info("Starting data validation")
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            logger.info(f"Created DVC tracking root: {self.data_validation_config.data_validation_dir}")

            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            # read data from train and test file path
            train_df=DataValidation.read_data(train_file_path)
            test_df=DataValidation.read_data(test_file_path)

            # 1️⃣ Schema checks for Train and Test
            def validate_dataset(df, name):
                valid = True
                if not self.validate_number_of_columns(df):
                    validation_error_msg = f"Incorrect column count in {name} dataset."
                    valid = False
                elif not self.is_columns_exist(df):
                    validation_error_msg = f"Missing columns in {name} dataset."
                    valid = False
                else:
                    validation_error_msg = f"{name} dataset passed schema validation."
                return valid, validation_error_msg
            
            train_valid, train_msg = validate_dataset(train_df, "train")
            test_valid, test_msg = validate_dataset(test_df, "test")
            validation_error_msg += f"\n{train_msg}\n{test_msg}"

            final_status = train_valid and test_valid 

            # Create output directories
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)

            valid_train_path = self.data_validation_config.valid_train_file_path
            valid_test_path = self.data_validation_config.valid_test_file_path
            invalid_train_path = self.data_validation_config.invalid_train_file_path
            invalid_test_path = self.data_validation_config.invalid_test_file_path
            
            # Save valid/invalid data accordingly
            if train_valid:
                train_df.to_csv(valid_train_path, index=False)
                logger.info(f"Train data saved at: {valid_train_path}")
            else:
                train_df.to_csv(invalid_train_path, index=False)
                logger.warning(f"Invalid Train data saved at: {invalid_train_path}")

            if test_valid:
                test_df.to_csv(valid_test_path, index=False)
                logger.info(f"Test data saved at: {valid_test_path}")
            else:
                test_df.to_csv(invalid_test_path, index=False)
                logger.warning(f"Invalid Test data saved at: {invalid_test_path}")

            # Data Drift Check (run even if drift exists)
            drift_summary = self.detect_dataset_drift(reference_df=train_df, current_df=test_df)
            drift_msg = " | No Data Drift Detected." if not drift_summary else f" | Data Drift Detected."
            validation_error_msg += drift_msg
            logger.info(drift_msg)

            data_validation_artifact = DataValidationArtifact(
            validation_status=final_status,
            valid_train_file_path=valid_train_path if train_valid else None,
            valid_test_file_path=valid_test_path if test_valid else None,
            invalid_train_file_path=None if train_valid else invalid_train_path,
            invalid_test_file_path=None if test_valid else invalid_test_path,
            message=validation_error_msg.strip(),
            drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        
        except Exception as e:
            raise CustomException(e)

if __name__=="__main__":
    try:
        logger.info("Starting Data Validation component execution")
        config=DataValidationConfig()
        data_ingestion_artifact=DataIngestionArtifact(
                                        train_file_path="artifacts/data_ingestion/ingested/train.csv", 
                                        test_file_path="artifacts/data_ingestion/ingested/test.csv")
        data_validation= DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                        data_validation_config=config)
        data_validation.initiate_data_validation()
        logger.info(f"Data Validation component finished successfully.")
        
    except Exception as e:
        logger.error(f"Data Validation component failed! Error: {e}")
        raise CustomException(e)
        
    
    

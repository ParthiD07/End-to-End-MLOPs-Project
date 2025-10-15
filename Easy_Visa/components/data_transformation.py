import os
import numpy as np
import pandas as pd
from pandas import DataFrame

from imblearn.combine import SMOTETomek
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from datetime import datetime

from Easy_Visa.constants import TARGET_COLUMN,SCHEMA_FILE_PATH

from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger
from Easy_Visa.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from Easy_Visa.entity.config_entity import DataTransformationConfig
from Easy_Visa.utils.main_utils import save_object,save_numpy_array,read_yaml


class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_transformation_config=data_transformation_config
            self._schema_config=read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e)
        
    @staticmethod
    def read_data(file_path)->DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e)
        
    @staticmethod
    def transform_target_column(target_series:pd.Series)->pd.Series:
        """Convert target labels to numeric"""
        try:
            mapping = {'Certified': 1, 'Denied': 0}
            return target_series.map(mapping)
        except Exception as e:
            raise CustomException(e)

    def get_data_transformer_object(self)-> make_pipeline:
        try:
            logger.info("Creating preprocessing pipeline for numeric, ordinal, and one-hot features.")

            oh_features = self._schema_config['oh_features']
            or_features = self._schema_config['or_features']
            num_features = self._schema_config['num_features']

            # Define order for ordinal encoding
            education_order = [['High School', "Bachelor's", "Master's", 'Doctorate']]

            # Pipelines for each data type
            numeric_pipeline = make_pipeline(StandardScaler())
            ordinal_pipeline = make_pipeline(OrdinalEncoder(categories=education_order))
            onehot_pipeline = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
            
            # Combine all pipelines
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_pipeline, num_features),
                ('ord', ordinal_pipeline, or_features),
                ('oh', onehot_pipeline, oh_features)
            ])
            logger.info("Preprocessing pipelines created successfully.")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e)


    def initiate_data_transformation(self)->DataTransformationArtifact:
        
        try:
            logger.info("Starting data validation")

            os.makedirs(self.data_transformation_config.data_transformation_dir,exist_ok=True)
            logger.info(f"Created DVC tracking root: {self.data_transformation_config.data_transformation_dir}")

            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # split independent and dependent columns on train data
            input_features_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df=train_df[TARGET_COLUMN]
            
            input_features_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            logger.info("Split train/test datasets into input and target features.")

            # Handle negative employee counts
            input_features_train_df["no_of_employees"] = input_features_train_df["no_of_employees"].abs()
            input_features_test_df["no_of_employees"] = input_features_test_df["no_of_employees"].abs()

            logger.info("Converted negative values in 'no_of_employees' column to absolute.")

            # Add company_age
            current_year = datetime.now().year
            input_features_train_df["company_age"] = current_year - input_features_train_df["yr_of_estab"]
            input_features_test_df["company_age"] = current_year - input_features_test_df["yr_of_estab"]

            logger.info("Added 'company_age' feature to both train and test sets.")

            # Drop unnecessary columns
            drop_cols = self._schema_config['drop_columns']
            input_features_train_df = input_features_train_df.drop(columns=drop_cols, axis=1)
            input_features_test_df = input_features_test_df.drop(columns=drop_cols, axis=1)
        

            # Encode target column
            target_feature_train_df = self.transform_target_column(target_feature_train_df)
            target_feature_test_df = self.transform_target_column(target_feature_test_df)


            # Get preprocessing object
            preprocessor = self.get_data_transformer_object()

            # Transform input features
            input_features_train_arr = preprocessor.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor.transform(input_features_test_df)

            smt=SMOTETomek(random_state=42)

            input_features_train_final, target_features_train_final = smt.fit_resample(
                    input_features_train_arr,target_feature_train_df)
            logger.info("Applied SMOTETomek on training dataset")

            # Combine transformed features with target column
            train_arr = np.c_[input_features_train_final, np.array(target_features_train_final)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            # Save artifacts
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, test_arr)

            logger.info("Data transformation completed and artifacts saved successfully.")

            # Return DataTransformationArtifact
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            
        except Exception as e:
            raise CustomException(e)

if __name__=="__main__":
    try:
        logger.info("Starting Data Transformation component execution")
        config=DataTransformationConfig()
        data_validation_artifact=DataValidationArtifact(
                                validation_status=True,
                                valid_train_file_path="artifacts/data_validation/validated/train.csv",
                                valid_test_file_path="artifacts/data_validation/validated/test.csv",
                                invalid_train_file_path=None,
                                invalid_test_file_path=None,
                                drift_report_file_path="artifacts/data_validation/drift_report/report.yaml")
        data_transformation= DataTransformation(data_validation_artifact=data_validation_artifact,
                                                data_transformation_config=config)
                                                
        data_transformation.initiate_data_transformation()
        logger.info(f"Data Transformation component finished successfully.")
        
    except Exception as e:
        logger.error(f"Data Transformation component failed! Error: {e}")
        raise CustomException(e)



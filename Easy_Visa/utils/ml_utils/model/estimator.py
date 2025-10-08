from Easy_Visa.constants import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os,sys
from pandas import DataFrame
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger

class VisaModel:
    def __init__(self,preprocessor,model):
        self.preprocessor=preprocessor
        self.model=model
    def predict(self,dataframe:DataFrame):
        try:
            x_transform=self.preprocessor.transform(dataframe)
            y_pred=self.model.predict(x_transform)
            return y_pred
        except Exception as e:
            raise CustomException(e)
        


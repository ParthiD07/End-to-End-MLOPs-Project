from pandas import DataFrame
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger

class VisaModel:
    def __init__(self,preprocessing_object, model):
        self.preprocessor=preprocessing_object
        self.model=model
    def predict(self,dataframe:DataFrame):
        try:
            logger.info("Using the trained model to get predictions")
            transformed_feature =self.preprocessor.transform(dataframe)
            y_pred=self.model.predict(transformed_feature)
            return y_pred
        except Exception as e:
            raise CustomException(e)


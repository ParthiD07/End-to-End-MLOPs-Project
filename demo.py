from Easy_Visa.logging.logger import logger
from Easy_Visa.exception.exception import CustomException

logger.info("welcome to custom logging")

try:
    a=1/0
except Exception as e:
    raise CustomException(e)

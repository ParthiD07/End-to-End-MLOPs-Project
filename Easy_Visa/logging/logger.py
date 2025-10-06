import logging # Python’s built-in logging library, used to record messages (info, warnings, errors) to files or console. It’s more flexible than print()
import os # Provides functions for interacting with the operating system (like creating folders, joining paths).
import sys
from datetime import datetime 

# --- Constants for Configuration ---
LOGGING_STR = "[%(asctime)s: %(levelname)s: %(module)s: %(funcName)s: %(lineno)d: %(message)s]"

# Define log directory path
log_dir="logs"

# Create directory if it doesn’t exist
os.makedirs(log_dir,exist_ok=True)

# Create a timestamped log file for each run
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
LOG_FILE_PATH=os.path.join(log_dir,f"running_logs_{timestamp}.log")

logging.basicConfig(
    format=LOGGING_STR,
    level=logging.INFO,
    handlers=[
        # 1. File Handler: Writes all logs to the persistent file
        logging.FileHandler(LOG_FILE_PATH),
        
        # 2. Stream Handler: Outputs logs to the console (standard output)
        logging.StreamHandler(sys.stdout)
    ]
)

logger= logging.getLogger(__name__)
logger.info(f"Logging started. Logs are saved to {LOG_FILE_PATH}")
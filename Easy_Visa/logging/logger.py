import logging # Python’s built-in logging library, used to record messages (info, warnings, errors) to files or console. It’s more flexible than print()
import os # Provides functions for interacting with the operating system (like creating folders, joining paths).
import sys

# --- Constants for Configuration ---
LOGGING_STR = "[%(asctime)s: %(levelname)s: %(module)s: %(funcName)s: %(lineno)d: %(message)s]"

# Define log directory path
log_dir="logs"

# Create directory if it doesn’t exist
os.makedirs(log_dir,exist_ok=True)

# Final log file path
LOG_FILE_PATH=os.path.join(log_dir,"running_logs.log")

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

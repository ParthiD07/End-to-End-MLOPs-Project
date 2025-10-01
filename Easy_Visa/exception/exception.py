import sys  # sys module gives access to system-specific functions, including sys.exc_info()
from Easy_Visa.logging.logger import logger  # custom logger (must exist in your project)


class CustomException(Exception):
    """
    Custom exception with detailed error info (file name, line number, type, message).
    """

    def __init__(self, error: Exception):
        # Call the parent Exception constructor with the error message
        super().__init__(str(error))

        # Get the exception details: (exception type, exception object, traceback object)
        _, _, exc_tb = sys.exc_info()

        # Traverse to the last traceback frame (deepest function call where error occurred)
        while exc_tb and exc_tb.tb_next:
            exc_tb = exc_tb.tb_next

        # Extract the file name where the error occurred (or Unknown if not available)
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown"

        # Extract the line number where the error occurred (or Unknown if not available)
        self.line_no = exc_tb.tb_lineno if exc_tb else "Unknown"

        # Get the type of the exception (e.g., ZeroDivisionError, ValueError, etc.)
        self.error_type = type(error).__name__

        # Get the actual error message (e.g., "division by zero")
        self.error_message = str(error)

        # Construct a detailed error message string
        self.full_message = (
            f"[{self.error_type}] occurred in script [{self.file_name}] "
            f"at line [{self.line_no}] -> {self.error_message}"
        )

        # Try logging the error with your project logger
        try:
            logger.error(self.full_message)
        except Exception:
            # If logging fails (e.g., logger not configured), fallback to printing
            print(self.full_message)

    def __str__(self):
        # When you print the exception, return the detailed full_message
        return self.full_message
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact: 
    """DataIngestionArtifact is a container (a dataclass or simple object) 
    that holds the file paths so later pipeline steps (like Data Validation) can use them"""
    train_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    message: str
    drift_report_file_path: str

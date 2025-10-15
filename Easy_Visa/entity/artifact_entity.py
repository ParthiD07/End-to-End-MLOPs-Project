from dataclasses import dataclass
from typing import Optional

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
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_file_path: str
    test_metric_file_path: str

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    s3_model_path: str
    trained_model_path: str
    evaluation_metrics_path: str

@dataclass
class ModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
    model_version: str
    pushed_at: str





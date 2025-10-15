import os,sys
import json
from typing import Dict, Any
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger

from Easy_Visa.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from Easy_Visa.entity.config_entity import ModelTrainerConfig

from Easy_Visa.utils.main_utils import save_object,load_object,load_numpy_array
from Easy_Visa.utils.ml_utils.model.estimator import VisaModel
from Easy_Visa.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier)
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn
import dagshub
from dagshub.common.errors import DagsHubRepoNotFoundError
import matplotlib.pyplot as plt
import io # For saving plots to MLflow


class ModelTrainer:

    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_config
        except Exception as e:
            raise CustomException(e)
        
    def get_base_models_and_params(self):
        """Defines models and hyperparameter grids (Moved from train_model)."""
        models = {
            "Random_forest": RandomForestClassifier(random_state=42),
            "Gradient_boosting": GradientBoostingClassifier(random_state=42),
            "Logistic_Regression": LogisticRegression(solver='liblinear', random_state=42),
            "Adaboost": AdaBoostClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)}
        
        params = {
            "Random_forest": {"n_estimators": [25,50,75,100], "max_depth": [5, 10, 15], "min_samples_split": [2, 5, 10]},
            "Gradient_boosting": {"n_estimators": [25,50,75,100], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7, 9]},
            "Logistic_Regression": {"penalty": ['l1', 'l2'], "C": [0.01, 0.1, 1.0]},
            "Adaboost": {"n_estimators": [25,50,75,100], "learning_rate": [0.01, 0.05, 0.1], "algorithm": ['SAMME', 'SAMME.R']},
            "XGBoost": {"n_estimators": [25,50,75,100], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7]}
        }
        return models, params
        
    
    def tune_and_log_models(self,x_train,y_train,x_test,y_test)-> Dict[str, Any]:
        try:
            report={}

            models, params = self.get_base_models_and_params()
            for name, model in models.items():
                # Start a new run for each model being tuned
                # This will be a nested run if initiate_model_trainer wraps everything in one run
                with mlflow.start_run(run_name=f"Tuning_{name}", nested=True) as run:
                    logger.info(f"Tuning and evaluating model: {name}")

                    # Get parameter grid for the model
                    param_grid = params.get(name, {})

                    # Log the parameter search space for this model
                    mlflow.log_params({f"param_search_{k}": str(v) for k, v in param_grid.items()})

                    # Perform hyperparameter tuning
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid,
                        n_iter=5,
                        cv=3,
                        scoring='f1',
                        n_jobs=-1,
                        random_state=42,
                        verbose=0)
                    
                    search.fit(x_train,y_train)

                    # Store and evaluate the best model
                    best_tuned_model = search.best_estimator_

                    # Log best parameters for the main run of this model
                    mlflow.log_params(search.best_params_)
                
                    # predictions
                    y_train_pred=best_tuned_model.predict(x_train)
                    y_test_pred=best_tuned_model.predict(x_test)

                    # F1 score
                    train_f1= f1_score(y_train,y_train_pred)
                    test_f1= f1_score(y_test,y_test_pred)

                    # Log F1 scores for model comparison
                    mlflow.log_metrics({"train_f1": train_f1, "test_f1": test_f1})

                    # Save results in report
                    report[name] = {
                        "model": best_tuned_model,
                        "train_f1": train_f1,
                        "test_f1": test_f1,
                        "best_params": search.best_params_,
                        "mlflow_run_id": run.info.run_id # Store run ID for later reference
                    }

                logger.info(
                    f"{name}: Train F1 = {train_f1:.3f}, Test F1 = {test_f1:.3f} | Best Params: {search.best_params_}"
                )

            return report

        except Exception as e:
            raise CustomException(e)
        
    def log_best_model_details(self, best_model, x_train, y_train, x_test, y_test):

        """Logs detailed final metrics, CM, and model to the main MLflow run."""
        try:
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            
            # 1. Log Detailed Classification Metrics
            mlflow.log_metrics({
                "final_train_f1": train_metric.f1_score,
                "final_train_precision": train_metric.precision_score,
                "final_train_recall": train_metric.recall_score,
                "final_test_f1": test_metric.f1_score,
                "final_test_precision": test_metric.precision_score,
                "final_test_recall": test_metric.recall_score,
                "f1_score_difference": abs(train_metric.f1_score - test_metric.f1_score)
            })

            # 2. Log Confusion Matrix
            cm = confusion_matrix(y_test, y_test_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(ax=ax)
            
            # Log the plot to MLflow
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            # 3. Overfitting/Underfitting Check
            diff = abs(train_metric.f1_score - test_metric.f1_score)
            mlflow.log_metric("f1_score_difference", diff)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logger.warning(
                    f"Possible Overfitting/Underfitting. F1 Diff: {diff:.4f} > Threshold: {self.model_trainer_config.overfitting_underfitting_threshold}"
                )

            return train_metric, test_metric

        except Exception as e:
            raise CustomException(e)
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logger.info("Starting Model Trainer and MLflow main run...")

            # Initialize Dagshub (use environment variable for secure URI)
            # Initialize Dagshub (use environment variable for secure URI)
            dagshub_uri = os.getenv("DAGSHUB_URI")

            if dagshub_uri:
                mlflow.set_tracking_uri(dagshub_uri)

            dagshub.init(
                repo_owner=os.getenv("DAGSHUB_USER"),
                repo_name=os.getenv("DAGSHUB_REPO"),
                mlflow=True
            )

            mlflow.set_experiment("EasyVisa_Model_Trainer")

            # Start the main MLflow run for the entire ModelTrainer component
            with mlflow.start_run(run_name="Model_Selection_and_Final_Training") as main_run:
            
                # --- Data Loading ---
                train_file_path=self.data_transformation_artifact.transformed_train_file_path
                test_file_path=self.data_transformation_artifact.transformed_test_file_path
            
                train_arr=load_numpy_array(train_file_path)
                test_arr=load_numpy_array(test_file_path)

                x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
                x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

                # --- Model Training and Hyperparameter Tuning ---
                report= self.tune_and_log_models(x_train,y_train,x_test,y_test)

                # --- Select Best Model ---
                best_model_name = max(report, key=lambda k: report[k]["test_f1"])
                best_model_metrics = report[best_model_name]
                best_model = best_model_metrics["model"]

                logger.info(f"Best model found: {best_model_name} | Train F1: {best_model_metrics['train_f1']:.3f} | Test F1: {best_model_metrics['test_f1']:.3f}")
                
                # Log best model details in the main run
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_metrics({
                    "best_train_f1": best_model_metrics["train_f1"],
                    "best_test_f1": best_model_metrics["test_f1"]
                })
                mlflow.log_params({f"best_param_{k}": v for k, v in best_model_metrics["best_params"].items()})
            
                # --- Final Model Validation ---
                if best_model_metrics["test_f1"] < self.model_trainer_config.expected_accuracy:
                    raise CustomException(
                        f"Best model {best_model_name} failed to meet the expected F1 threshold of {self.model_trainer_config.expected_accuracy}. "
                        f"Achieved only {best_model_metrics['test_f1']:.3f}"
                    )
                logger.info(f"Model passed expected accuracy threshold: {best_model_metrics['test_f1']:.3f} >= {self.model_trainer_config.expected_accuracy}")

                # --- Final Evaluation and Logging (Best Practice) ---
                train_metric, test_metric = self.log_best_model_details(best_model, x_train, y_train, x_test, y_test)

                # Load preprocessor and save trained model
                preprocessor=load_object(self.data_transformation_artifact.transformed_object_file_path)

                model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
                os.makedirs(model_dir_path,exist_ok=True)

                # Create the final VisaModel object
                visa_model=VisaModel(preprocessing_object=preprocessor,model=best_model)
                # Save the model object locally (for the pipeline)
                save_object(self.model_trainer_config.trained_model_file_path,model=visa_model)

                # --- Save Metrics as JSON ---
                metrics_dir = os.path.join(model_dir_path, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)

                train_metric_path = os.path.join(metrics_dir, "train_metrics.json")
                test_metric_path = os.path.join(metrics_dir, "test_metrics.json")

                # Convert train/test metric objects to dict and save
                with open(train_metric_path, "w") as f:
                    json.dump(train_metric.__dict__, f, indent=4)

                with open(test_metric_path, "w") as f:
                    json.dump(test_metric.__dict__, f, indent=4)

                mlflow.log_artifact(train_metric_path, artifact_path="metrics")
                mlflow.log_artifact(test_metric_path, artifact_path="metrics")

                # Set a consistent registered model name
                registered_model_name = "EasyVisa_Classifier"

                # Log the VisaModel to MLflow/DagsHub and register it
                mlflow.sklearn.log_model(
                    sk_model=visa_model, 
                    artifact_path="visa_model_artifact", 
                    registered_model_name=registered_model_name
                )
                logger.info(f"Model logged and registered as '{registered_model_name}' on MLflow/DagsHub.")

                # Create and return artifact
                model_trainer_artifact=ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    train_metric_file_path=train_metric_path,
                    test_metric_file_path=test_metric_path)
            
                logger.info(f"Model Trainer completed successfully. Model saved at: {self.model_trainer_config.trained_model_file_path}")

                return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e)
        

if __name__=="__main__":
    try:
        logger.info("Starting Model Trainer component execution")
        config=ModelTrainerConfig()
        data_transformation_artifact=DataTransformationArtifact(
            transformed_object_file_path="artifacts/data_transformation/transformed_object/preprocessing.pkl",
            transformed_train_file_path="artifacts/data_transformation/transformed/train.npy",
            transformed_test_file_path="artifacts/data_transformation/transformed/test.npy"
        )

        model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                   model_trainer_config=config)
                                                
        model_trainer.initiate_model_trainer()
        logger.info(f"Model Trainer component finished successfully.")
        
    except Exception as e:
        logger.error(f"Model Trainer component failed! Error: {e}")
        raise CustomException(e)

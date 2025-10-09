import os,sys
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
from sklearn.metrics import f1_score

import mlflow
import mlflow.sklearn
import dagshub


class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_config
        except Exception as e:
            raise CustomException(e)
        
    
    def tune_and_log_models(self,x_train,y_train,x_test,y_test,models,params):
        try:
            report={}

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

                    # Log all hyperparameter search results (optional but good practice)
                    results = search.cv_results_
                    for i in range(len(results['params'])):
                        with mlflow.start_run(run_name=f"Trial_{i}", nested=True):
                            mlflow.log_params(results['params'][i])
                            mlflow.log_metric("mean_cv_f1", results['mean_test_score'][i])

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


    def train_model(self,x_train,y_train,x_test,y_test):
        """
        Trains multiple models and returns their evaluation report.
        """
        models={
            "Random_forest": RandomForestClassifier(),
            "Gradient_boosting": GradientBoostingClassifier(),
            "Logistic_Regression": LogisticRegression(),
            "Adaboost": AdaBoostClassifier(),
            "XGBoost": XGBClassifier()}
        
        params = {
    "Random_forest": {
        "n_estimators": [25,50,75,100],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4, 8]
    },

    "Gradient_boosting": {
        "n_estimators": [25,50,75,100],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 9],
        # "subsample": [0.8, 0.9, 1.0]
    },

    "Logistic_Regression": {
        "penalty": ['l1', 'l2'],
        "C": [0.01, 0.1, 1.0],
        "solver": ['liblinear'],
        # "max_iter": [25,50,75,100]
    },

    "Adaboost": {
        "n_estimators": [25,50,75,100],
        "learning_rate": [0.01, 0.05, 0.1],
        "algorithm": ['SAMME', 'SAMME.R'],
        # "estimator": [DecisionTreeClassifier(max_depth=d) for d in [1, 2, 3, 4]]
    },

    "XGBoost": {
        "n_estimators": [25,50,75,100],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        # "subsample": [0.7, 0.8, 0.9, 1.0]
    }}
        
        model_report=self.tune_and_log_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                          models=models,params=params)
        
        return model_report
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logger.info("Starting Model Trainer and MLflow main run...")

            # Initialize Dagshub (use environment variable for secure URI)
            dagshub_uri = os.getenv("DAGSHUB_URI")
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
                report= self.train_model(x_train,y_train,x_test,y_test)

                # --- Select Best Model ---
                best_model_name = max(report, key=lambda k: report[k]["test_f1"])
                best_model_metrics = report[best_model_name]
                best_model = best_model_metrics["model"]
                best_params = best_model_metrics["best_params"]

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
                        f"Best model {best_model_name} failed to meet the expected F1 threshold of {self.expected_accuracy}. "
                        f"Achieved only {best_model_metrics['test_f1']:.3f}"
                    )
                logger.info(f" Model passed expected accuracy threshold: {best_model_metrics['test_f1']:.3f} >= {self.model_trainer_config.expected_accuracy}")

                 # --- Final Evaluation (using comprehensive metrics) ---
                y_train_pred=best_model.predict(x_train)
                y_test_pred = best_model.predict(x_test)

                train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
                test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

                # Log detailed final metrics in the main run
                mlflow.log_metrics({
                    "final_train_f1": train_metric.f1_score,
                    "final_train_precision": train_metric.precision_score,
                    "final_train_recall": train_metric.recall_score,
                    "final_test_f1": test_metric.f1_score,
                    "final_test_precision": test_metric.precision_score,
                    "final_test_recall": test_metric.recall_score
                })

                # Load preprocessor and save trained model
                preprocessor=load_object(self.data_transformation_artifact.transformed_object_file_path)

                model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
                os.makedirs(model_dir_path,exist_ok=True)

                # Create the final VisaModel object
                visa_model=VisaModel(preprocessor=preprocessor,model=best_model)
                # Save the model object locally (for the pipeline)
                save_object(self.model_trainer_config.trained_model_file_path,model=visa_model)

                # Log the VisaModel to MLflow/DagsHub
                # This automatically stores the model file and logs it to the DVC remote specified in DagsHub
                mlflow.sklearn.log_model(visa_model, "visa_model")

                 # --- Overfitting/Underfitting check ---
                diff = abs(train_metric.f1_score - test_metric.f1_score)
                mlflow.log_metric("f1_score_difference", diff)
                if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                    logger.warning(
                        f"Possible Overfitting/Underfitting. F1 Diff: {diff:.4f} > Threshold: {self.model_trainer_config.overfitting_underfitting_threshold}"
                    )

                # Create and return artifact
                model_trainer_artifact=ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                        train_metric_arifact=train_metric,
                                                        test_metric_arifact=test_metric)
            
                logger.info(f"Model Trainer completed successfully. Model saved at: {self.model_trainer_config.trained_model_file_path}")

                return model_trainer_artifact
        
        except Exception as e:
            raise CustomException(e)
        

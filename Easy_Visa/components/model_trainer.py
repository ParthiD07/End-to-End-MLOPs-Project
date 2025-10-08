import os,sys
from Easy_Visa.exception.exception import CustomException
from Easy_Visa.logging.logger import logger

from Easy_Visa.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from Easy_Visa.entity.config_entity import ModelTrainerConfig

from Easy_Visa.utils.main_utils import save_object,load_object,load_numpy_array,evaluate_models
from Easy_Visa.utils.ml_utils.model.estimator import VisaModel
from Easy_Visa.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier)
from xgboost import XGBClassifier

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        try:
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_config=model_trainer_config
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
        
        model_report=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                          models=models,params=params)
        
        return model_report, models

        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logger.info("Starting Model Trainer...")
            
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            #loading training and test array
            train_arr=load_numpy_array(train_file_path)
            test_arr=load_numpy_array(test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            report, models= self.train_model(x_train,y_train,x_test,y_test)

            best_model_name = max(report,key=lambda k:report[k]["test_f1"])

            best_model_metrics = report[best_model_name]

            best_model= models[best_model_name]

            logger.info(f"Best model found: {best_model_name} | Train F1: {best_model_metrics['train_f1']:.3f} | Test F1: {best_model_metrics['test_f1']:.3f}")
            
            # Check if model meets expected accuracy
            if best_model_metrics["test_f1"] < self.model_trainer_config.expected_accuracy:
                raise CustomException(
                    f"Best model {best_model_name} failed to meet the expected F1 threshold of {self.expected_accuracy}. "
                    f"Achieved only {best_model_metrics['test_f1']:.3f}"
                )
            logger.info(f" Model passed expected accuracy threshold: {best_model_metrics['test_f1']:.3f} >= {self.model_trainer_config.expected_accuracy}")

            # Evaluate the chosen model
            best_model.fit(x_train, y_train)
            y_train_pred=best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
            test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

            # Load preprocessor and save trained model
            preprocessor=load_object(self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            visa_model=VisaModel(preprocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,model=visa_model)

            # Overfitting/Underfitting check
            if abs(train_metric.f1_score - test_metric.f1_score) > self.model_trainer_config.overfitting_underfitting_threshold:
                logger.warning("Possible Overfitting or Underfitting detected.")

            # Create and return artifact
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                    train_metric_arifact=train_metric,
                                                    test_metric_arifact=test_metric)
            
            logger.info(f"Model Trainer completed successfully. Model saved at: {self.model_trainer_config.trained_model_file_path}")

            return model_trainer_artifact
        
                                           
        except Exception as e:
            raise CustomException(e)
        

import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logging import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:

            logging.info("Splitting the train and test data inputs!")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbor Classifer" : KNeighborsRegressor(),
                "XG Boost Classifier" : XGBRegressor(),
                "CatBoost Classifier" : CatBoostRegressor(verbose=False),
                "ADA Boost Regressor" : AdaBoostRegressor()
            }

            model_reports : dict = evaluate_models(X_train = x_train, y_train= y_train, X_test = x_test, y_test=y_test, models= models)

            best_model_score = max(sorted(model_reports.values()))
            best_model_name = list(model_reports.keys())[
                list(model_reports.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model if found !")
            logging.info("Best model found for both training and the testing !")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(x_test)
            r2_scoree = r2_score(y_test, predicted)

            return r2_scoree
        except Exception as e:
            raise CustomException(e,sys)
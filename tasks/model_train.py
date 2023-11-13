import json
from typing import Annotated

import mlflow
import pandas as pd
from prefect import get_run_logger, task
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from models import model_dev

# from tasks.config import logger, PARAMETERS_PATH
from tasks.config import PARAMETERS_PATH


def load_parametres() -> dict:
    """
    Load parameters from a JSON file and log it as an artifact in MLflow.

    Returns:
        dict: A dictionary containing the loaded parameters.
    """
    parameters_path = PARAMETERS_PATH
    f = open(parameters_path)
    parameters = json.load(f)
    mlflow.log_artifact(parameters_path, artifact_path="parameters")

    return parameters


@task(log_prints=True)
def model_train_best_parameters(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, "model"]:
    """
    Trains a machine learning model using the best parameters found through RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The target variable for the training data.

    Returns:
        Pipeline: The trained machine learning model pipeline.
    """

    mlflow.start_run()
    logger = get_run_logger()
    # Read parametres
    parameters = load_parametres()

    # pipline
    full_processor = model_dev.data_pipline(X_train)

    model_select = model_dev.model_select("XGBRegressor")
    model_grid = RandomizedSearchCV(estimator=model_select, **parameters)
    model_pipeline = Pipeline(
        steps=[("preprocess", full_processor), ("model", model_grid)]
    )
    model = model_pipeline.fit(X_train, y_train)

    logger.info(f"Best params are: {model['model'].best_params_}")
    mlflow.log_params(model["model"].best_params_)
    mlflow.sklearn.log_model(model, artifact_path="models_mlflow")

    return model

import mlflow
import numpy as np
import pandas as pd

# from tasks.config import logger
from prefect import get_run_logger, task
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


@task(log_prints=True)
def evaluation(X_test: pd.DataFrame, y_test: pd.Series, model: Pipeline) -> None:
    logger = get_run_logger()

    predictions = model.predict(X_test)

    logger.info(f"fit_model.best_params: {model['model'].best_params_}")

    # Evaluate the model

    # mse
    mse = mean_squared_error(y_test, predictions)
    logger.info("The mean squared error value is: " + str(mse))
    mlflow.log_metric("mse", mse)
    # r2_score
    r2 = r2_score(y_test, predictions)
    logger.info("The r2 score value is: " + str(r2))
    mlflow.log_metric("r2", r2)
    # rmse
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    logger.info("The root mean squared error value is: " + str(rmse))
    mlflow.log_metric("rmse", rmse)
    # mlflow.end_run()

    # pathlib.Path("models").mkdir(exist_ok=True)
    logger.info(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

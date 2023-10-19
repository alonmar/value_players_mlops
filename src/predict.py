from typing import Dict

import mlflow
import numpy as np
import pandas as pd

TRACKING_URL = "http://127.0.0.1:5000"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model(run_id: str):
    """Load model by run_id

    Args:
        run_id (str): id mlflow

    Returns:
        _type_: _description_
    """
    logged_model = f"runs:/{run_id}/models_mlflow"
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def human_readable_payload(value_predict, run_id) -> Dict:
    """Takes numpy array and returns back human readable dictionary

    Args:
        value_predict (_type_): _description_
        run_id (_type_): _description_

    Returns:
        Dict: _description_
    """

    value_log = float(np.round(value_predict, 2))
    value_stimate = float(np.round(np.exp(value_predict), 2))
    result = {
        "value_log": value_log,
        "value_money": f"{value_stimate} euros",
        "run_id": run_id,
    }
    return result


def predict(pX: dict, run_id: str) -> Dict:
    """Takes weight and predicts height

    Args:
        pX (dict): _description_
        run_id (str): _description_

    Returns:
        Dict: _description_
    """

    model = load_model(run_id)
    df = pd.DataFrame(pX)
    prediction = model.predict(df)

    result = human_readable_payload(prediction, run_id)
    return result

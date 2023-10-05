import numpy as np
import mlflow
import pandas as pd

TRACKING_URL = "http://127.0.0.1:5000"

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

mlflow.set_tracking_uri(TRACKING_URL)


def load_model(run_id):
    logged_model = f'runs:/{run_id}/models_mlflow'
    model = mlflow.pyfunc.load_model(logged_model)
    return model


def human_readable_payload(value_predict):
    """Takes numpy array and returns back human readable dictionary"""

    value_log = float(np.round(value_predict, 2))
    value_stimate = float(np.round(np.exp(value_predict), 2))
    result = {
        "value_log": value_log,
        "value_money": f"{value_stimate} euros",
    }
    return result


def predict(pX: dict, run_id: str) -> dict:
    """Takes weight and predicts height"""

    model = load_model(run_id)
    df = pd.DataFrame(pX)
    prediction = model.predict(df)

    result = human_readable_payload(prediction)
    return result

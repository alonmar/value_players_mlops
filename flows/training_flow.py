from prefect import flow

from tasks.clean_data import clean_data
from tasks.evaluation import evaluation
from tasks.ingest_data import ingest_data
from tasks.model_train import model_train_best_parameters


@flow
def train_pipeline() -> None:
    """
    Trains a machine learning model using the following steps:
    1. Ingests data using the `ingest_data` function.
    2. Cleans the data using the `clean_data` function.
    3. Trains the model using the `model_train_best_parameters` function.
    4. Evaluates the model using the `evaluation` function.
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = model_train_best_parameters(x_train, y_train)
    evaluation(x_test, y_test, model)

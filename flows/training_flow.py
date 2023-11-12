from tasks.clean_data import clean_data
from tasks.evaluation import evaluation
from tasks.ingest_data import ingest_data
from tasks.model_train import model_train_best_parameters
from prefect import flow


@flow
def train_pipeline() -> None:
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        mse: float
        rmse: float
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = model_train_best_parameters(x_train, y_train)
    evaluation(x_test, y_test, model)

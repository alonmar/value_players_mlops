import mlflow
import typer

from flows.training_flow import train_pipeline
from monitoring.dummy_metrics_calculation import dummy_metrics_monitoring

app = typer.Typer()


@app.command()
def train():
    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("xgb_reg")

    train_pipeline()

    print(
        "************************************************************************\n"
        "Now run \n"
        f"    mlflow ui --backend-store-uri '{mlflow.get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`\n"
        "experiment. Here you'll also be able to compare the two runs.)\n"
        "************************************************************************\n"
    )


@app.command()
def dummy_metrics():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_name = "xgb"
    alias = "champion"

    dummy_metrics_monitoring(model_name, alias)


if __name__ == "__main__":
    app()

import mlflow

from flows.training_flow import train_pipeline

if __name__ == "__main__":
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

import json

# logging
import logging
import pathlib

import numpy as np
import mlflow

# Data management
import pandas as pd
from prefect import flow, task

# Models
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split

logging.basicConfig(level=logging.INFO)


@task(retries=3, retry_delay_seconds=2)
def read_data() -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_json("data_raw/data_fifa_players.json")
    return df


def change_data_types(df):
    """change data types for better manage"""

    # cm
    df["height"] = df["height"].str.extract("(\\d+)").astype(int)
    # kg
    df["weight"] = df["weight"].str.extract("(\\d+)").astype(int)
    # euros
    df["Value"] = pd.to_numeric(df["Value"].str.replace("€|\\.", "", regex=True))
    df["Wage"] = pd.to_numeric(df["Wage"].str.replace("€|\\.", "", regex=True))

    return df


def get_position_zone(df):
    """Transform 'preferred_positions' to 'position_zone'"""
    position_zone = []
    for x in df["preferred_positions"]:
        listb = {"DEFENDING": 0, "MIDFIELD": 0, "ATTACKING": 0, "GOALKEEPER": 0}
        for y in x:
            if y in ["GK"]:
                listb["GOALKEEPER"] = 1
            else:
                if y in ["LF", "RF", "CF", "ST"]:
                    listb["ATTACKING"] = listb["ATTACKING"] + 1
                if y in ["CAM", "RM", "RW", "CDM", "CM", "LM", "LW"]:
                    listb["MIDFIELD"] = listb["MIDFIELD"] + 1
                if y in ["LB", "LWB", "RB", "RWB", "CB"]:
                    listb["DEFENDING"] = listb["DEFENDING"] + 1

        # Keep the position with the highest value
        position_zone.append(max(listb, key=listb.get))

    df.loc[:, "position_zone"] = position_zone
    df = df.drop(columns=["preferred_positions"])

    return df


def one_hot_coding(df):
    """Conver columns type 'object' to one hot coding, then
    delete leaving only the One hot coding"""
    dummies_object = None
    columns_type_object = []

    for i, column_type in enumerate([str(d) for d in df.dtypes]):
        if column_type == "object":
            column_name = df.columns[i]
            columns_type_object.append(column_name)

            dummies = pd.get_dummies(df[column_name], prefix=column_name)

            if dummies_object is None:
                dummies_object = dummies
            else:
                dummies_object = pd.concat([dummies_object, dummies], axis=1)

    df = df.drop(columns_type_object, axis="columns")
    df = pd.concat([df, dummies_object], axis=1)
    return df


@task
def clean_data(df):
    df = df.drop(columns=["url"])
    df = df.dropna().copy()
    df = change_data_types(df)
    df = get_position_zone(df)
    df = df.drop(columns=["Wage", "Birth Date", "name", "nation"])
    df = one_hot_coding(df)

    return df


@task
def split_data(df, tsize=0.2):
    y = df["Value"].values  # Target
    # Transform to log
    y = np.log(y)
    y = y.reshape(-1, 1)
    X = df.drop(columns=["Value"])  # Feature(s)
    scaler = StandardScaler()
    X_scaler = scaler.fit(X)
    X = X_scaler.transform(X)
    y_scaler = scaler.fit(y)
    y = y_scaler.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tsize, random_state=3
    )

    return X_train, X_test, y_train, y_test, y_scaler


@task(log_prints=True)
def train_best_model(X_train, X_test, y_train, y_test, y_scaler) -> None:
    """train a model with best hyperparams and write everything out"""

    with mlflow.start_run():
        # parameters = {
        #    "param_distributions": {
        #        "max_depth": [3, 6, 10],
        #        "learning_rate": [0.01, 0.05, 0.1, 0.3, 0.5],
        #        "n_estimators": [100, 500, 1000],
        #        "colsample_bytree": [0.7, 0.5],
        #    },
        #    "scoring": "neg_mean_squared_error",
        #    "verbose": 3,
        #    "n_jobs": -1,
        # }

        # Read parametres
        f = open("parametres.json")
        parameters = json.load(f)

        mlflow.log_artifact("parametres.json", artifact_path="parametres")

        xgb_model = XGBRegressor(seed=20)
        xgb_grid = RandomizedSearchCV(estimator=xgb_model, **parameters)
        model = xgb_grid.fit(X_train, y_train)
        mlflow.log_params(model.best_params_)

        # accuracy = model.score(X_test, y_test)
        predictions = model.predict(X_test)

        logging.debug(f"fit_model.best_params: {model.best_params_}")
        rmse_xgb_reg = mean_squared_error(
            y_scaler.inverse_transform(y_test),
            y_scaler.inverse_transform(predictions.reshape(-1, 1)),
            squared=False,
        )

        # logging(f"The RMSE for xgb_reg is: {rmse_xgb_reg}")
        # logging(f"Best params are: {model.best_params_}")

        mlflow.log_metric("rmse", rmse_xgb_reg)

        pathlib.Path("models").mkdir(exist_ok=True)

        mlflow.sklearn.log_model(model, artifact_path="models_mlflow")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")
    return None


@flow
def main_flow() -> None:
    """The main training pipeline"""

    # MLflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("xgb_reg")

    # Load
    df = read_data()

    # Transform

    df = clean_data(df)

    X_train, X_test, y_train, y_test, y_scaler = split_data(
        df,
        tsize=0.2,
    )

    # Train
    train_best_model(X_train, X_test, y_train, y_test, y_scaler)


if __name__ == "__main__":
    main_flow()

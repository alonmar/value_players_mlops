import json
from typing import Dict

import mlflow
import numpy as np

# Data management
import pandas as pd
from prefect import flow, task
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Models
# pylint: disable=no-name-in-module
from xgboost import XGBRegressor

from src import config

# logging


@task(retries=3, retry_delay_seconds=2)
def read_data() -> pd.DataFrame:
    """Read data into DataFrame

    Returns:
        pd.DataFrame: _description_
    """
    df = pd.read_json("data_raw/data_fifa_player_train.json")
    df = df.drop(columns=["url"])
    return df


def change_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """

    # cm
    df["height"] = df["height"].str.extract("(\\d+)").astype(int)
    # kg
    df["weight"] = df["weight"].str.extract("(\\d+)").astype(int)
    # euros
    df["Value"] = pd.to_numeric(df["Value"].str.replace("€|\\.", "", regex=True))
    df["Wage"] = pd.to_numeric(df["Wage"].str.replace("€|\\.", "", regex=True))

    return df


def get_position_zone(df: pd.DataFrame) -> pd.DataFrame:
    """Transform 'preferred_positions' to 'position_zone'

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
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


def one_hot_coding(df: pd.DataFrame) -> pd.DataFrame:
    """Conver columns type 'object' to one hot coding, then
    delete leaving only the One hot coding

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
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
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna().copy()
    df = change_data_types(df)
    df = get_position_zone(df)
    df = df.drop(columns=["Wage", "Birth Date", "name", "nation"])
    df = one_hot_coding(df)

    return df


@task
def split_data(df: pd.DataFrame, tsize=0.2) -> list:
    y = df["Value"].values  # Target
    # Transform to log
    y = np.log(y)
    y = y.reshape(-1, 1)
    X = df.drop(columns=["Value"])  # Feature(s)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, random_state=3)

    return X_train, X_test, y_train, y_test


def data_pipline(X_train: pd.DataFrame) -> ColumnTransformer:
    # Para el escalado de las variables numericas usaremos MinMaxScaler
    # y como lo mensionamos anterirormente usaremos el promedio para la imputación
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="mean")),
            ("StandardScaler", StandardScaler()),
            ("scale", MinMaxScaler()),
        ]
    )
    # handle_unknown='ignore' es importante en caso de tomar una categoria que no se encontraba
    # durante el proceso de entrenamiento
    # pylint: disable=line-too-long
    categorical_pipeline = Pipeline(steps=[("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    # diferenciamos varibles numericas y las que no los son
    numerical_features = X_train.select_dtypes(include=["number"]).columns
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns
    full_processor = ColumnTransformer(
        transformers=[
            ("number", numeric_pipeline, numerical_features),
            ("category", categorical_pipeline, categorical_features),
        ]
    )

    return full_processor


def load_parametres() -> Dict:
    parameters_path = "parametres.json"
    f = open(parameters_path)
    parameters = json.load(f)
    mlflow.log_artifact(parameters_path, artifact_path="parametres")
    return parameters


@task(log_prints=True)
def train_best_model(X_train, X_test, y_train, y_test) -> None:
    """train a model with best hyperparams and write everything out

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_

    Returns:
        _type_: _description_
    """

    with mlflow.start_run():
        # Read parametres
        parameters = load_parametres()

        # pipline
        full_processor = data_pipline(X_train)

        xgb_model = XGBRegressor(seed=20)
        xgb_grid = RandomizedSearchCV(estimator=xgb_model, **parameters)
        xgb_pipeline = Pipeline(steps=[("preprocess", full_processor), ("model", xgb_grid)])
        model = xgb_pipeline.fit(X_train, y_train)
        mlflow.log_params(model["model"].best_params_)
        # accuracy = model.score(X_test, y_test)
        predictions = model.predict(X_test)
        config.logger.debug(f"fit_model.best_params: {model['model'].best_params_}")
        rmse_xgb_reg = mean_squared_error(
            y_test,
            predictions.reshape(-1, 1),
            squared=False,
        )

        config.logger.debug(f"The RMSE for xgb_reg is: {rmse_xgb_reg}")
        config.logger.debug(f"Best params are: {model['model'].best_params_}")

        mlflow.log_metric("rmse", rmse_xgb_reg)

        # pathlib.Path("models").mkdir(exist_ok=True)

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

    X_train, X_test, y_train, y_test = split_data(
        df,
        tsize=0.2,
    )

    # Train
    train_best_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main_flow()

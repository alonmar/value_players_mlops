# Models
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


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
    categorical_pipeline = Pipeline(
        steps=[("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False))]
    )
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


def model_select(model_name: str) -> RegressorMixin:
    if model_name == "XGBRegressor":
        model = XGBRegressor(seed=20)
    else:
        raise ValueError("Model name not supported")

    return model

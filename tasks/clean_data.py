from prefect import task
import pandas as pd
from models import data_cleaning
from typing import Tuple
from typing_extensions import Annotated


@task(log_prints=True)
def clean_data(
    df: pd.DataFrame, tsize=0.2
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    df = df.dropna().copy()
    df = data_cleaning.change_data_types(df)
    df = data_cleaning.get_position_zone(df)
    df = df.drop(columns=["Wage", "Birth Date", "name", "nation"])
    df = data_cleaning.one_hot_coding(df)
    X_train, X_test, y_train, y_test = data_cleaning.split_data(
        df,
        tsize=tsize,
    )

    return X_train, X_test, y_train, y_test
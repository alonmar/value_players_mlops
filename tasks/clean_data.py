from typing import Annotated

import pandas as pd
from prefect import task

from models import data_cleaning


@task(log_prints=True)
def clean_data(
    df: pd.DataFrame, tsize=0.2
) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    This function receives a pandas DataFrame and a test size value, performs some data cleaning operations,
    splits the data into training and testing sets, and returns the resulting DataFrames and Series.

    Args:
    - df: pandas DataFrame containing the data to be cleaned and split.
    - tsize: float value representing the proportion of the data to be used for testing. Default is 0.2.

    Returns:
    - tuple containing the following DataFrames and Series:
        - X_train: pandas DataFrame containing the training data.
        - X_test: pandas DataFrame containing the testing data.
        - y_train: pandas Series containing the target values for the training data.
        - y_test: pandas Series containing the target values for the testing data.
    """

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

# Data management
from typing import Annotated

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def change_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes the data types of the columns "height", "weight", "Value" and "Wage" in the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be modified.

    Returns:
        pd.DataFrame: The modified DataFrame.
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
    """
    Given a DataFrame with a column "preferred_positions" containing a list of strings representing soccer positions,
    this function returns a new DataFrame with a column "position_zone" containing a string representing the zone of the
    player based on their preferred positions. The zones are "DEFENDING", "MIDFIELD", "ATTACKING" and "GOALKEEPER".
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
    """
    Applies one-hot encoding to all columns of type 'object' in a pandas DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be encoded.

    Returns:
        pd.DataFrame: The encoded DataFrame.
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


def split_data(
    df: pd.DataFrame, tsize: float
) -> tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Split the input DataFrame into training and testing sets.

    Args:
        df: pandas DataFrame with the data to split.
        tsize: float representing the proportion of the data to include in the test split.

    Returns:
        tuple with the following elements:
            - X_train: pandas DataFrame with the training features.
            - X_test: pandas DataFrame with the testing features.
            - y_train: pandas Series with the training target.
            - y_test: pandas Series with the testing target.
    """

    y = df["Value"].values  # Target
    # Transform to log
    y = np.log(y)
    y = y.reshape(-1, 1)
    X = df.drop(columns=["Value"])  # Feature(s)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tsize, random_state=3
    )

    return X_train, X_test, y_train, y_test

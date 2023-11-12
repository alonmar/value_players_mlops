# Data management
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from typing_extensions import Annotated


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


def split_data(
    df: pd.DataFrame, tsize: float
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    y = df["Value"].values  # Target
    # Transform to log
    y = np.log(y)
    y = y.reshape(-1, 1)
    X = df.drop(columns=["Value"])  # Feature(s)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=tsize, random_state=3
    )

    return X_train, X_test, y_train, y_test

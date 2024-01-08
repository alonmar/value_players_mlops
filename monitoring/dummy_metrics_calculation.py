import logging
import time
from datetime import datetime
from typing import Optional
import numpy as np
import mlflow
from models import data_cleaning
import pandas as pd
from sqlalchemy_utils import create_database, database_exists
from sqlmodel import (
    TIMESTAMP,
    Column,
    Field,
    Session,
    SQLModel,
    create_engine,
    text,
)


def dummy_metrics_monitoring(model_name: str, alias: str):
    loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")

    time.sleep(2.4)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
    )

    class Player(SQLModel, table=True):
        id: Optional[int] = Field(default=None, primary_key=True)
        predict: float = Field(default=None)
        real: float = Field(default=None)
        created_datetime: Optional[datetime] = Field(
            sa_column=Column(
                TIMESTAMP(timezone=True),
                nullable=False,
                server_default=text("CURRENT_TIMESTAMP"),
            )
        )

    SQLITE_URL = "postgresql://postgres:example@localhost:5432/test"

    if not database_exists(SQLITE_URL):
        create_database(SQLITE_URL)

    engine = create_engine(SQLITE_URL, echo=True)

    SQLModel.metadata.create_all(engine)

    SEND_TIMEOUT = 5

    df = pd.read_json("data/data_fifa_player_test.json")

    # Clean data
    df = df.dropna().copy()
    df = data_cleaning.change_data_types(df)
    df = data_cleaning.get_position_zone(df)
    df = df.drop(columns=["Wage", "Birth Date", "name", "nation"])
    df = data_cleaning.one_hot_coding(df)

    y = df["Value"].values  # Target
    # Transform to log
    y = np.log(y)
    y = y.reshape(-1, 1)
    X = df.drop(columns=["Value"])

    for i in range(1, len(df)):
        time.sleep(SEND_TIMEOUT)
        result = loaded_model.predict(X.iloc[[i]])
        logging.info(result)
        values = Player(predict=result, real=y[i][0])

        session = Session(engine)

        session.add(values)

        session.commit()
        logging.info("data sent")

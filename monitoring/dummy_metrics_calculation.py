import logging
import random
import time
from datetime import datetime
from typing import Optional

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

from src import config
from src.predict import predict

time.sleep(2.4)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


class Hero(SQLModel, table=True):
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


sqlite_url = "postgresql://postgres:example@localhost:5432/test"

if not database_exists(sqlite_url):
    create_database(sqlite_url)

engine = create_engine(sqlite_url, echo=True)

SQLModel.metadata.create_all(engine)


SEND_TIMEOUT = 10
rand = random.Random()


df = pd.read_json("data_raw/data_fifa_player_test.json")

run_id = config.RUN_ID

y = df["Value"].values  # Target
# Transform to log
# y = y.reshape(-1, 1)
X = df.drop(columns=["Value"])  # Feature(s)

for i in range(1, len(df)):
    time.sleep(5)

    result = predict(X.iloc[[i]], run_id, is_dict=False)
    print(result)
    values = Hero(predict=result["value_log"], real=y[i])

    session = Session(engine)

    session.add(values)

    session.commit()
    logging.info("data sent")

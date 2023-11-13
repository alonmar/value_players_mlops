import pandas as pd
from prefect import task


@task(retries=3, retry_delay_seconds=2)
def ingest_data() -> pd.DataFrame:
    """Read data into DataFrame

    Returns:
        pd.DataFrame: _description_
    """
    df = pd.read_json("data/data_fifa_player_train.json")
    df = df.drop(columns=["url"])
    return df

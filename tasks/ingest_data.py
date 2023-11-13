import pandas as pd
from prefect import task


@task(retries=3, retry_delay_seconds=2)
def ingest_data() -> pd.DataFrame:
    """
    Reads a JSON file containing FIFA player data and returns a pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing FIFA player data.
    """
    df = pd.read_json("data/data_fifa_player_train.json")
    df = df.drop(columns=["url"])
    return df

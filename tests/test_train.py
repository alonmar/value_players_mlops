import pandas as pd

from src import train


def test_get_position_zone():
    # intialise data of lists.
    data = {
        "preferred_positions": [["LF", "RF", "CAM"], ["CAM"], ["LWB", "RWB", "RB"]],
        "name": ["test1", "test2", "test3"],
    }
    # Create DataFrame
    df = pd.DataFrame(data)
    result = train.get_position_zone(df)
    # pylint: disable=unsubscriptable-object
    assert ["ATTACKING", "MIDFIELD", "DEFENDING"] == result["position_zone"].tolist()

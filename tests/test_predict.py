from src.predict import human_readable_payload


def test_human_readable_payload():
    # intialise data of lists.
    result = human_readable_payload(0.2, "xxxxxxxxx")
    assert 0.2 == result["value_log"]
    assert "1.22 euros" == result["value_money"]
    assert "xxxxxxxxx" == result["run_id"]

    result = human_readable_payload(0.7, "xxxxxxxxx")
    assert 0.7 == result["value_log"]
    assert "2.01 euros" == result["value_money"]
    assert "xxxxxxxxx" == result["run_id"]

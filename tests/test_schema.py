from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.schema import validate_raw


def good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dteday": pd.to_datetime(["2011-01-01", "2011-01-01"]),
            "hr": [0, 1],
            "season": [1, 1],
            "yr": [0, 0],
            "mnth": [1, 1],
            "holiday": [0, 0],
            "weekday": [6, 6],
            "workingday": [0, 0],
            "weathersit": [1, 2],
            "temp": [0.24, 0.22],
            "atemp": [0.2879, 0.2727],
            "hum": [0.81, 0.80],
            "windspeed": [0.0, 0.0],
            "cnt": [16, 40],
        }
    )


def test_valid_frame_passes() -> None:
    assert len(validate_raw(good_frame())) == 2


def test_hour_out_of_range_is_rejected() -> None:
    frame = good_frame()
    frame.loc[0, "hr"] = 25
    with pytest.raises(SchemaErrors):
        validate_raw(frame)


def test_negative_target_is_rejected() -> None:
    frame = good_frame()
    frame.loc[1, "cnt"] = -5
    with pytest.raises(SchemaErrors):
        validate_raw(frame)


def test_intentional_failure():
    assert 1 == 2, 'Intentionally failing test for CI pipeline test'

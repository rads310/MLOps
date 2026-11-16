from __future__ import annotations

import numpy as np
import pandas as pd

from src.featurize import build
from src.prepare import split


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instant": [1, 2, 3, 4],
            "dteday": pd.to_datetime(["2011-01-01"] * 4),
            "dteday_hour": pd.to_datetime(
                [
                    "2011-01-01 00:00",
                    "2011-01-01 06:00",
                    "2011-01-01 12:00",
                    "2011-01-01 18:00",
                ]
            ),
            "hr": [0, 6, 12, 18],
            "temp": [0.2, 0.3, 0.4, 0.5],
            "casual": [1, 2, 3, 4],
            "registered": [10, 20, 30, 40],
            "cnt": [11, 22, 33, 44],
        }
    )


def test_leakage_columns_are_dropped() -> None:
    built = build(
        sample_frame(),
        target="cnt",
        drop=["instant", "dteday", "casual", "registered"],
        cyclic_hour=True,
    )
    assert "casual" not in built.columns
    assert "registered" not in built.columns
    assert built.columns[-1] == "cnt"


def test_cyclic_hour_is_continuous() -> None:
    built = build(
        sample_frame(), target="cnt", drop=["instant", "dteday"], cyclic_hour=True
    )
    assert {"hr_sin", "hr_cos"} <= set(built.columns)
    assert np.allclose(built["hr_sin"] ** 2 + built["hr_cos"] ** 2, 1.0)


def test_time_split_keeps_order() -> None:
    train, test = split(sample_frame(), test_size=0.25, by_time=True, seed=42)
    assert train["dteday_hour"].max() <= test["dteday_hour"].min()

from __future__ import annotations

import pandas as pd

from src.predict import prepare_features

PARAMS = {
    "data": {"target": "cnt", "drop": ["instant", "dteday", "casual", "registered"]},
    "featurize": {"cyclic_hour": True},
}
FEATURES = ["season", "yr", "mnth", "holiday", "weekday", "hr_sin", "hr_cos"]


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instant": [1, 2],
            "dteday": pd.to_datetime(["2012-01-01", "2012-01-01"]),
            "hr": [0, 12],
            "season": [1, 1],
            "yr": [1, 1],
            "mnth": [1, 1],
            "holiday": [0, 0],
            "weekday": [0, 0],
        }
    )


def test_columns_match_model_order() -> None:
    prepared = prepare_features(raw_frame(), PARAMS, FEATURES)
    assert list(prepared.columns) == FEATURES


def test_missing_feature_becomes_empty_column() -> None:
    prepared = prepare_features(raw_frame(), PARAMS, [*FEATURES, "temp"])
    assert prepared["temp"].isna().all()

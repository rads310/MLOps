from __future__ import annotations
import pandera as pa, pandas as pd

_raw_schema = pa.DataFrameSchema({
    "instant": pa.Column(int, nullable=False),
    "dteday": pa.Column(pa.DateTime, nullable=False),
    "season": pa.Column(int, pa.Check.isin([1, 2, 3, 4])),
    "yr": pa.Column(int, pa.Check.isin([0, 1])),
    "mnth": pa.Column(int, pa.Check.between(1, 12)),
    "hr": pa.Column(int, pa.Check.between(0, 23)),
    "holiday": pa.Column(int, pa.Check.isin([0, 1])),
    "weekday": pa.Column(int, pa.Check.between(0, 6)),
    "workingday": pa.Column(int, pa.Check.isin([0, 1])),
    "weathersit": pa.Column(int, pa.Check.between(1, 4)),
    "temp": pa.Column(float, pa.Check.between(0, 1)),
    "atemp": pa.Column(float, pa.Check.between(0, 1)),
    "hum": pa.Column(float, pa.Check.between(0, 1)),
    "windspeed": pa.Column(float, pa.Check.between(0, 1)),
    "casual": pa.Column(int, nullable=False),
    "registered": pa.Column(int, nullable=False),
    "cnt": pa.Column(int, pa.Check.ge(0), nullable=False),
}, coerce=True)

def validate_raw(frame: pd.DataFrame) -> pd.DataFrame:
    return _raw_schema.validate(frame)

"""Схема исходных данных. Проверка выполняется до любой обработки."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

RAW_SCHEMA = DataFrameSchema(
    {
        "dteday": Column(pa.DateTime, nullable=False),
        "hr": Column(int, Check.in_range(0, 23), nullable=False),
        "season": Column(int, Check.in_range(1, 4), nullable=False),
        "yr": Column(int, Check.in_range(0, 1), nullable=False),
        "mnth": Column(int, Check.in_range(1, 12), nullable=False),
        "holiday": Column(int, Check.isin([0, 1]), nullable=False),
        "weekday": Column(int, Check.in_range(0, 6), nullable=False),
        "workingday": Column(int, Check.isin([0, 1]), nullable=False),
        "weathersit": Column(int, Check.in_range(1, 4), nullable=False),
        "temp": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "atemp": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "hum": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "windspeed": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        "cnt": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    },
    strict=False,
    coerce=True,
    name="raw_hourly",
)


def validate_raw(frame):
    """Проверяет таблицу целиком и собирает все нарушения (lazy=True)."""
    return RAW_SCHEMA.validate(frame, lazy=True)

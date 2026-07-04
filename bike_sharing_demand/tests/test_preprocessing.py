"""
Tests for the preprocessing module.
"""

import pandas as pd

from src_bike_sharing.preprocessing import add_datetime_features, prepare_features


def test_hour_is_extracted_correctly():
    """A datetime at 08:00 should produce hour == 8."""
    df = pd.DataFrame({
        "datetime": ["2011-01-01 08:00:00"],
    })

    result = add_datetime_features(df)

    assert result["hour"].iloc[0] == 8


def test_all_datetime_features_are_added():
    """add_datetime_features should create hour, dayofweek, month, year."""
    df = pd.DataFrame({
        "datetime": ["2011-01-01 08:00:00"],
    })

    result = add_datetime_features(df)

    for feature in ["hour", "dayofweek", "month", "year"]:
        assert feature in result.columns


def test_leakage_columns_dropped_from_train():
    """casual and registered must be removed when is_train=True."""
    df = pd.DataFrame({
        "datetime": ["2011-01-01 08:00:00"],
        "casual": [3],
        "registered": [13],
        "count": [16],
        "temp": [9.84],
    })

    result = prepare_features(df, is_train=True)

    assert "casual" not in result.columns
    assert "registered" not in result.columns


def test_raw_datetime_column_dropped():
    """The raw datetime string should be gone after prepare_features."""
    df = pd.DataFrame({
        "datetime": ["2011-01-01 08:00:00"],
        "temp": [9.84],
    })

    result = prepare_features(df, is_train=False)

    assert "datetime" not in result.columns
"""
Tests for the preprocessing module.

Each test builds a tiny, hand-made DataFrame with a known problem
(a missing value, a text column) and asserts that the preprocessing
function fixes it correctly. Small controlled inputs make failures
easy to diagnose.
"""

import numpy as np
import pandas as pd

from src.preprocessing import fill_missing_values, encode_categorical


def test_fill_missing_leaves_no_nulls():
    """After filling, there should be zero missing values anywhere."""
    df = pd.DataFrame({
        "PoolQC": [np.nan, "Ex", np.nan],      # absent-feature text
        "GarageArea": [np.nan, 400, 500],      # absent-feature numeric
        "LotArea": [8000, np.nan, 9500],       # genuine numeric leftover
    })

    result = fill_missing_values(df)

    assert result.isnull().sum().sum() == 0


def test_fill_missing_uses_none_for_absent_features():
    """Missing PoolQC should become the string 'None', not something else."""
    df = pd.DataFrame({
        "PoolQC": [np.nan, "Ex"],
        "LotArea": [8000, 9000],
    })

    result = fill_missing_values(df)

    assert result["PoolQC"].iloc[0] == "None"


def test_fill_missing_uses_zero_for_absent_numeric():
    """Missing GarageArea should become 0."""
    df = pd.DataFrame({
        "GarageArea": [np.nan, 400],
        "LotArea": [8000, 9000],
    })

    result = fill_missing_values(df)

    assert result["GarageArea"].iloc[0] == 0


def test_encode_removes_all_text_columns():
    """After encoding, no column should still be text (object) type."""
    df = pd.DataFrame({
        "MSZoning": ["RL", "RM", "RL"],
        "LotArea": [8000, 9000, 9500],
    })

    result = encode_categorical(df)

    text_columns = result.select_dtypes(include="str").columns
    assert len(text_columns) == 0


def test_encode_does_not_mutate_original():
    """Encoding must not change the DataFrame that was passed in."""
    df = pd.DataFrame({"MSZoning": ["RL", "RM"]})

    encode_categorical(df)  # call it, ignore the result

    # original should STILL be text — proof we didn't mutate it
    assert not pd.api.types.is_numeric_dtype(df["MSZoning"])
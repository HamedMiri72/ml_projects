"""
Preprocessing module.

Handles feature engineering and preparation:
  1. Extracts time-based features from the datetime column
  2. Drops leakage columns (casual, registered) and the raw datetime
  3. Separates features from the target

The key work here is datetime feature engineering: the raw timestamp
is turned into hour / day-of-week / month / year, which capture the
strong daily and seasonal demand cycles.
"""
from tkinter.messagebox import RETRY

import pandas as pd
from pandas.core.interchange import column
from pandas.core.interchange.dataframe_protocol import DataFrame

from bike_sharing_demand.src import config


def add_datetime_features(df: pd.DataFrame) -> DataFrame
    """
    Turn the raw datetime string into useful numeric features.

    hour       -> captures the twin commuter peaks (8am, 5-6pm)
    dayofweek  -> weekday vs weekend behaviour (0=Mon .. 6=Sun)
    month      -> seasonal trend across the year
    year       -> demand grew from 2011 to 2012
    """

    df = df.copy()

    dt = pd.to_datetime(df["datetime"])

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["dayofweek"] = dt.dt.dayofweek
    df["hour"] = dt.dt.hour

    return df

def prepare_features(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """
    Produce a model-ready feature DataFrame.

    Adds datetime features, then drops columns the model must not use:
      - the raw 'datetime' string (replaced by extracted features)
      - leakage columns (only on train; test doesn't have them)
      - the target itself (handled separately by the caller)
    """

    df = add_datetime_features(df)

    df = df.drop(columns=["datetime"])

    if is_train:
        df = df.drop(columns = config.LEAKAGE_COLUMNS)

    return df

def split_features_target(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
        Split the training DataFrame into features (X) and target (y).
    """
    y = train_df[config.TARGET]
    X = train_df.drop(columns=[config.TARGET])

    return X, y





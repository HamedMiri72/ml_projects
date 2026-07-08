"""
Preprocessing module.

Handles all data cleaning and feature preparation:
  1. Extract passenger's title from name column.
  2. Filling missing values (with domain-aware logic)
  3. Encoding categorical (text) columns into numbers

"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src_titanic import config


def add_title_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the passenger's title (Mr, Mrs, Miss, ...) from Name."""
    df = df.copy()

    # Regex: grab the word right before a period, after a comma+space
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    # Collapse rare titles into "Rare" so the model isn't confused by
    # one-off categories like "Countess", "Capt", "Col", etc.
    common = ["Mr", "Mrs", "Miss", "Master" ]
    df["Title"] = df["Title"].apply(lambda x: x if x in common else "Rare")
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the known gaps: Age, Embarked, Fare."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every text column into integer labels."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature prep: title, fills, encoding, drops."""
    df = add_title_feature(df)          # extract Title from Name
    df = fill_missing_values(df)        # fill gaps
    df = df.drop(columns=["Name"])      # drop Name (already mined)
    df = df.drop(columns=config.DROP_COLUMNS)  # drop Ticket/Cabin/PassengerId
    df = encode_categoricals(df)        # encode last, only on kept columns
    return df

def split_features_target(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split training data into features (X) and target (y)."""
    y = train_df[config.TARGET]
    X = train_df.drop(columns=[config.TARGET])
    return X, y











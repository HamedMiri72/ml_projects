"""
Preprocessing module.

Handles all data cleaning and feature preparation:
  1. Filling missing values (with domain-aware logic)
  2. Encoding categorical (text) columns into numbers

Design note: train and test are processed together to guarantee
identical treatment. The target (SalePrice) is never passed in here,
so there is no risk of leaking it into the features.
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas.core.interchange import column
from sklearn.preprocessing import LabelEncoder

# Columns where a missing value means "this feature is absent"
# (e.g. no pool), so we fill with the string "None" as a real category.
_NONE_COLUMNS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "MasVnrType",
]

# Numeric columns where a missing value means "zero of it"
# (e.g. no garage -> garage area is 0).
_ZERO_COLUMNS = [
    "GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
]

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill every missing value using domain-aware rules.

    Three cases:
      1. Absent-feature text columns -> "None"
      2. Absent-feature numeric columns -> 0
      3. Genuine leftovers -> most frequent (text) or median (numeric)

    """

    df = df.copy()


    # Case 1: absent feature, text -> "None"
    for col in _NONE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    #Case2: absent features, numeric -> 0
    for col in _ZERO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Case 3a: remaining text -> most frequent value
    for col in df.select_dtypes(include=["str"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Case 3b: remaining numeric -> most frequent value
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert every text column into integer labels.

    Label encoding is used (rather than one-hot) because the downstream
    model is tree-based, which handles integer-coded categories well.
    """
    df = df.copy()

    for col in df.select_dtypes(include=["str"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for both datasets.

    Combines train and test, cleans and encodes them identically,
    then splits them back apart. Drops the 'Id' column (not a feature).

    Returns:
        (X_train, X_test) — cleaned, encoded, ready for modeling.
    """
    n_train = len(train)

    # Combine so both sets get identical treatment
    combined = pd.concat([train,test], axis=0, ignore_index=True)

    combined = fill_missing_values(combined)
    combined = encode_categorical(combined)

    # Split back apart using the remembered training length
    X_train = combined.iloc[:n_train].copy()
    X_test = combined.iloc[n_train:].copy()

    # Id is a label, not a predictor
    X_train = X_train.drop(columns = "Id")
    X_test = X_test.drop(columns = "Id")

    return X_train, X_test



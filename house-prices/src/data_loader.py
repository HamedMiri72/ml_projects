

"""
Data loading module.

Responsible solely for reading raw CSV files from disk and returning
them as pandas DataFrames. No cleaning or transformation happens here —
that is the job of preprocessing.py.
"""

import pandas as pd
from src import config

def load_train() -> pd.DataFrame:
    """Load the raw training data (features + SalePrice)."""
    return pd.read_csv(config.TRAIN_FILE)

def load_test() -> pd.DataFrame:
    """Load the raw training data (features + SalePrice)."""
    return pd.read_csv(config.TEST_FILE)

def load_sample_submission() -> pd.DataFrame:
    """Load the sample submission, used for the correct Id ordering."""
    return pd.read_csv(config.SUBMISSION_FILE)
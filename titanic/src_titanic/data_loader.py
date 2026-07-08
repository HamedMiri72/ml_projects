
"""
Data loading module.

Responsible solely for reading raw CSV files from disk and returning
them as pandas DataFrames. No cleaning or transformation happens here —
that is the job of preprocessing.py.
"""

import pandas as pd
from src_titanic import config

def load_train() -> pd.DataFrame:
    """Load the training data and return it as a DataFrame."""
    return pd.read_csv(config.TRAIN_FILE)

def load_test() -> pd.DataFrame:
    """Load the testing data and return it as a DataFrame."""
    return pd.read_csv(config.TEST_FILE)

def load_sample_submission() -> pd.DataFrame:
    """Load the sample submission data and return it as a DataFrame."""
    return pd.read_csv(config.SAMPLE_SUBMISSION_FILE)

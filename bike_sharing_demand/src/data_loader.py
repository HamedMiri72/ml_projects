"""
Data loading module.

Reads the raw CSV files from disk and returns them as pandas DataFrames.
No cleaning or transformation happens here.
"""

import pandas as pd

from src import config

def load_train() -> pd.DataFrame:
    """Load the raw training data (features + count)."""
    return pd.read_csv(config.TRAIN_FILE)

def load_test() -> pd.DataFrame:
    """Load the row testing data (feature + count)."""
    return pd.read_csv(config.TEST_FILE)

def load_sample_submission() -> pd.DataFrame:
    """Load the sample submission, used for the correct datetime ordering."""
    return pd.read_csv(config.SAMPLE_SUBMISSION_FILE)



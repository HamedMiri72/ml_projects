"""
Prediction module.

Loads a trained model, predicts SalePrice on the test set, and writes
the Kaggle submission file in the required format (two columns: Id,
SalePrice).
"""


import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import config

def load_model() -> RandomForestRegressor:
    """Load the trained model from disk."""
    with open(config.MODEL_FILE, "rb") as f:
        return pickle.load(f)

def make_submission(model: RandomForestRegressor, X_test: pd.DataFrame, test_ids: pd.Series,) -> pd.DataFrame:
    """
    Predict on the test set and build the submission DataFrame.

    Args:
        model:    the trained regressor
        X_test:   preprocessed test features (no Id column)
        test_ids: the original Id column, for the submission's first column

    Returns:
        A DataFrame with exactly the columns Kaggle expects: Id, SalePrice.
    """
    predictions = model.predict(X_test)
    submission = pd.DataFrame({
        config.TARGET: predictions,
    })
    submission.insert(0, "Id", test_ids.values)
    submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
    return submission

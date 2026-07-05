"""
Prediction module.

Loads a trained model, predicts demand on the test set, and writes
the Kaggle submission file (two columns: datetime, count).
"""

import pickle

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src_bike_sharing import config


def load_model(model: RandomForestRegressor) -> None:

    with open(config.MODEL_FILE, "rb") as f:
        pickle.load(f)

def make_submission(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    test_datetimes: pd.Series,
) -> pd.DataFrame:
    """
    Predict on the test set and build the submission DataFrame.

    Args:
        model:          the trained regressor
        X_test:         preprocessed test features (no datetime column)
        test_datetimes: the original datetime strings, for the submission

    Returns:
        A DataFrame with the columns Kaggle expects: datetime, count.
    """
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, None)  # demand can't be negative

    submission = pd.DataFrame({
        "datetime": test_datetimes.values,
        config.TARGET: predictions,
    })

    submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
    return submission




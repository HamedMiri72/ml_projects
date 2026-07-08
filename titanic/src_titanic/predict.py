"""
Prediction module.

Loads a trained models, predicts Survived on the test set, and writes
the Kaggle submission file in the required format (two columns: PassengerId,
Survived).
"""


import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src_titanic import config

def load_model() -> RandomForestClassifier:
    """Load model from the path"""
    with open(config.MODEL_FILE, "rb") as f:
       return pickle.load(f)


def make_submission(model: RandomForestClassifier, X_test: pd.DataFrame, test_ids: pd.Series,) -> pd.DataFrame:
    """
        Predict on the test set and build the submission DataFrame.

        Args:
            model:    the trained Classifier
            X_test:   preprocessed test features (no Id column)
            test_ids: the original Id column, for the submission's first column

        Returns:
            A DataFrame with exactly the columns Kaggle expects: PassengerId, Survived.
        """
    predictions = model.predict(X_test)
    submission = pd.DataFrame({
        config.TARGET: predictions
    })

    submission.insert(0, "PassengerId", test_ids.values)
    submission.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
    return submission



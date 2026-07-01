"""
Training module.

Trains a Random Forest regressor on the preprocessed data, evaluates it
on a held-out validation split, and saves the final model to disk.

The evaluation metric is RMSE on log-transformed prices, matching the
Kaggle competition's scoring. Log scale is used because SalePrice is
right-skewed, so it keeps errors on cheap and expensive homes comparable.
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src import config

def rmse_log(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error on log prices."""
    return float(np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred))))


def evaluate_model(X: pd.DataFrame, y: pd.DataFrame) -> float:
    """
    Train on 80% and score on a held-out 20% to estimate real-world
    performance. Returns the validation RMSE (log scale).
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    model = RandomForestRegressor(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE)

    model.fit(X_tr, y_tr)

    preds = model.predict(X_val)
    score = rmse_log(y_val, preds)

    print(f"Validation RMSE (log scale): {score:.5f}")
    return score

def train_final_model(X: pd.DataFrame, y: pd.DataFrame) -> RandomForestRegressor:
    """
    Train the production model on ALL available data.

    Validation already told us the model is good; now we use every row
    so the final model is as informed as possible.
    """

    model = RandomForestRegressor(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE)
    model.fit(X, y)
    return model

def save_model(model: RandomForestRegressor) -> None:
    """Persist the trained model to disk with pickle."""
    config.MODELS_DIR.mkdir(exist_ok=True)  # create models/ if missing
    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {config.MODEL_FILE}")



"""
Training module.

Trains a Random Forest regressor to predict hourly bike demand,
evaluates it on a held-out split, and saves the final model.

The competition metric is RMSLE (Root Mean Squared Logarithmic Error).
Log scale is used because demand is right-skewed and because a given
absolute error matters more at low counts than at high ones.
"""
import pickle
from enum import CONFORM

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src_bike_sharing.preprocessing import add_datetime_features, prepare_features, split_features_target
from src_bike_sharing import config

def rmsle(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Logarithmic Error — the Kaggle metric.

    Uses log(1 + x) so that a value of 0 is handled safely
    (log(0) is undefined, but log(1 + 0) = 0).
    """
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    return float(np.sqrt(mean_squared_error(log_true, log_pred)))

def evaluate_model(X: pd.DataFrame, y: pd.Series) -> float:
    """Train on 80%, score on a held-out 20%. Returns validation RMSLE."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
    )
    model.fit(X_tr, y_tr)

    preds = model.predict(X_val)
    preds = np.clip(preds, 0, None)  # demand can't be negative

    score = rmsle(y_val, preds)
    print(f"Validation RMSLE: {score:.5f}")
    return score

def train_final_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """Train the production model on ALL available data."""
    model = RandomForestRegressor(
        n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE
    )
    model.fit(X, y)
    return model

def save_model(model: RandomForestRegressor) -> None:
    """Persist the trained model to disk with pickle."""
    config.MODELS_DIR.mkdir(exist_ok=True)
    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {config.MODEL_FILE}")





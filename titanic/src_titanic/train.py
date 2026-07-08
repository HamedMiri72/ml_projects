"""
Train module

Evaluate the model and return a score which shows the accuracy score of model
train full model on real data and full data available
save model into the models folder to reuse it later
"""
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src_titanic import config


def evaluate_model(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Train on 80% and score on a held-out 20% to estimate real-world
    performance.
    """

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    model = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE)

    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    score = accuracy_score(y_val, preds)
    print(f"Accuracy: {score * 100:.1f}%")
    return score

def train_final_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Train the final model on all available data."""
    model = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, random_state=config.RANDOM_STATE)
    model.fit(X, y)
    return model

def save_model(model: RandomForestClassifier) -> None:
    """Persist the trained models to disk with pickle."""
    config.MODELS_DIR.mkdir(exist_ok=True)  # create models/ if missing
    with open(config.MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {config.MODEL_FILE}")


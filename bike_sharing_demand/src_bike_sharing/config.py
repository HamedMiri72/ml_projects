"""
Central configuration for the bike-sharing demand project.

All paths, filenames, and model hyperparameters live here so nothing
is hardcoded in the logic.
"""

from pathlib import Path

# ---- Project paths ----
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ---- Data files ----
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sampleSubmission.csv"   # note: camelCase!
SUBMISSION_FILE = PROJECT_ROOT / "submission.csv"

# ---- Model file ----
MODEL_FILE = MODELS_DIR / "bike_demand_model.pkl"

# ---- The column we're predicting ----
TARGET = "count"

# ---- Columns to drop before training ----
# casual + registered only exist in train and sum to count -> data leakage.
LEAKAGE_COLUMNS = ["casual", "registered"]

# ---- Model hyperparameters ----
RANDOM_STATE = 42
N_ESTIMATORS = 200
TEST_SIZE = 0.2
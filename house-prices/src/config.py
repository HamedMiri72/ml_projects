"""
Central configuration for the house-prices project.

All paths, filenames, and model hyperparameters live here so that
nothing is hardcoded inside the logic. Change a setting once, here,
and it applies everywhere.
"""

from pathlib import Path

# ---- Project paths ----
# Path(__file__) is THIS file's location. .parent.parent climbs up
# from src/config.py to the project root, so paths work on any machine.
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# ---- Data files ----
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "sample_submission.csv"
SUBMISSION_FILE = PROJECT_ROOT / "submission.csv"

# ---- Model file ----
MODEL_FILE = MODELS_DIR / "house_price_model.pkl"

# ---- The column we're predicting ----
TARGET = "SalePrice"

# ---- Model hyperparameters ----
# Kept here so tuning is one edit, not a hunt through the code.
RANDOM_STATE = 42       # fixed seed = reproducible results every run
N_ESTIMATORS = 200      # number of trees in the random forest
TEST_SIZE = 0.2         # fraction held out for validation
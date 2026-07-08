"""
Central configuration for the titanic project.

All paths, filenames, and models hyperparameters live here so that
nothing is hardcoded inside the logic. Change a setting once, here,
and it applies everywhere.
"""

from pathlib import Path

# ---- Project paths ----
# Path(__file__) is THIS file's location. .parent.parent climbs up
# from src_titanic/config.py to the project root, so paths work on any machine.
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


# ---- Data file paths ----
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_FILE = DATA_DIR / "gender_submission.csv"

# ---- Submission file ----
SUBMISSION_FILE = PROJECT_ROOT / "submission.csv"

# ---- Model file ----
MODEL_FILE = MODELS_DIR / "titanic_model.pkl"

# ---- The column we're predicting ----
TARGET = "Survived"

# ---- Model hyperparameters ----
# Kept here so tuning is one edit, not a hunt through the code.
RANDOM_STATE = 42       # fixed seed = reproducible results every run
N_ESTIMATORS = 200      # number of trees in the random forest
TEST_SIZE = 0.2         # fraction held out for validation

# ---- Columns to DROP ----
DROP_COLUMNS = ["PassengerId", "Ticket", "Cabin"]

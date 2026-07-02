"""
Central configuration for the BIKE-SHARING-DEMAND project.

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
TRAIN_DIR = DATA_DIR / "train.csv"
TEST_DIR = DATA_DIR / "test.csv"
SAMPLE_SUUBMISSION_FILE = DATA_DIR / "sampleSubmission.csv"
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



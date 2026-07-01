# House Prices — Advanced Regression

An end-to-end machine learning pipeline that predicts residential home sale prices from 79 explanatory features, built for the [Kaggle House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

The focus of this project is a clean, modular, production-style codebase: separated concerns, a central configuration, and unit-tested preprocessing.

## Results

| Metric | Score |
| --- | --- |
| Validation RMSE (log scale) | ~0.152 |

The evaluation metric is Root Mean Squared Error on log-transformed sale prices, matching the competition's scoring. The log transform keeps errors on inexpensive and expensive homes comparable, since sale price is right-skewed.

## Project structure

​​```text
house-prices/
├── data/                    # raw CSVs (not tracked in git)
├── models/                  # trained model artifacts (not tracked)
├── src/
│   ├── config.py            # all paths and hyperparameters
│   ├── data_loader.py       # reads raw CSVs
│   ├── preprocessing.py     # missing-value handling + encoding
│   ├── train.py             # training and evaluation
│   └── predict.py           # submission generation
├── tests/
│   └── test_preprocessing.py
├── main.py                  # runs the full pipeline
├── requirements.txt
└── README.md
​```

## Approach

1. **Data loading** — raw train and test CSVs are read from `data/`.
2. **Preprocessing** — train and test are combined so both receive identical treatment, then split back apart. Missing values are filled with domain-aware rules (absent features become `"None"` or `0`; genuine gaps use median or mode). Categorical columns are label-encoded, which suits the tree-based model.
3. **Modelling** — a Random Forest regressor is evaluated on a held-out 20% split, then retrained on all data for the final model.
4. **Prediction** — the trained model generates the Kaggle submission.

## Setup

​```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
​```

Download the competition data from Kaggle and place `train.csv`, `test.csv`, and `sample_submission.csv` in the `data/` directory.

## Usage

​```bash
python main.py
​```

This trains the model, saves it to `models/`, and writes `submission.csv` in the project root, ready to upload to Kaggle.

## Testing

​```bash
python -m pytest tests/ -v
​```

## Tech stack

Python, pandas, NumPy, scikit-learn, pytest.

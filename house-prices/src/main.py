"""
House Prices — end-to-end training pipeline.

Runs the full workflow in order:
    1. Load raw data
    2. Preprocess (clean + encode)
    3. Evaluate the model on a validation split
    4. Train the final model on all data and save it
    5. Generate the Kaggle submission file

Run from the project root with:
    python main.py
"""

from src import data_loader, preprocessing, predict, train, config


def run_pipeline() -> None:
    """Execute the complete house-prices pipeline end to end."""

    # --- 1. Load raw data ---
    print("Loading data...")
    train_df = data_loader.load_train()
    test_df = data_loader.load_test()

    # Separate the target from the training features
    y = train_df[config.TARGET]
    train_features = train_df.drop(columns=[config.TARGET])

    # Remember the test Ids for the submission (before Id gets dropped)
    test_ids = test_df["Id"]

    # --- 2. Preprocess ---
    print("Preprocessing...")
    X_train, X_test = preprocessing.preprocess(train_features, test_df)

    # --- 3. Evaluate ---
    print("Evaluating model on validation split...")
    train.evaluate_model(X_train, y)

    # --- 4. Train final model on all data and save ---
    print("Training final model on all data...")
    model = train.train_final_model(X_train, y)
    train.save_model(model)

    # --- 5. Generate submission ---
    print("Generating submission...")
    predict.make_submission(model, X_test, test_ids)

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
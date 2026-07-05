"""
Bike Sharing Demand — end-to-end training pipeline.

Runs the full workflow in order:
    1. Load raw data
    2. Capture test datetimes (needed for the submission)
    3. Preprocess (datetime features, drop leakage/raw columns)
    4. Evaluate the model on a validation split
    5. Train the final model on all data and save it
    6. Generate the Kaggle submission file

Run from the project root with:
    python main.py
"""

from src_bike_sharing import data_loader, preprocessing, predict, train

def run_pipeline() -> None:
    """Execute the complete bike-sharing pipeline end to end."""

    # --- 1. Load raw data ---
    print("Loading data...")
    train_df = data_loader.load_train()
    test_df = data_loader.load_test()

    # --- 2. Capture test datetimes BEFORE preprocessing drops them ---
    test_datetimes = test_df["datetime"]

    # --- 3. Preprocess ---
    print("Preprocessing...")
    train_prepared = preprocessing.prepare_features(train_df, is_train=True)
    X_test = preprocessing.prepare_features(test_df, is_train=False)
    X_train, y = preprocessing.split_features_target(train_prepared)

    # --- 4. Evaluate ---
    print("Evaluating model on validation split...")
    train.evaluate_model(X_train, y)

    # --- 5. Train final model on all data and save ---
    print("Training final model on all data...")
    model = train.train_final_model(X_train, y)
    train.save_model(model)

    # --- 6. Generate submission ---
    print("Generating submission...")
    predict.make_submission(model, X_test, test_datetimes)

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
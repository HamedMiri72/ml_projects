"""
Titanic — end-to-end training pipeline.

Runs the full workflow in order:
    1. Load raw data
    2. Preprocess (clean + encode)
    3. Evaluate the models on a validation split
    4. Train the final models on all data and save it
    5. Generate the Kaggle submission file

Run from the project root with:
    python main.py
"""

from src_titanic import data_loader, preprocessing, predict, train

def run_pipeline() -> None:
    """Execute the complete titanic pipeline end to end."""

    # --- 1. Load raw data ---
    print("Loading data...")
    train_df = data_loader.load_train()
    test_df = data_loader.load_test()

    # --- 2. Capture test IDs BEFORE preprocessing drops PassengerId ---
    test_ids = test_df["PassengerId"]

    # --- 3. Preprocess (train and test separately) ---
    print("Preprocessing...")
    train_prepared = preprocessing.prepare_features(train_df)
    X_test = preprocessing.prepare_features(test_df)
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
    predict.make_submission(model, X_test, test_ids)

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()


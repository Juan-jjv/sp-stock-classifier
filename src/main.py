import os
import pandas as pd

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    RESULTS_PATH,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    LOOKAHEAD_DAYS,
    TEST_SIZE_RATIO
)

from load_data import load_spx_data
from feature_engineering import create_model_dataset
from train_models import (
    split_time_series_data,
    train_models,
    evaluate_models
)


def main():
    print("Loading raw data...")
    raw_df = load_spx_data(RAW_DATA_PATH)

    print("Raw data loaded successfully.")
    print(raw_df.head())
    print(f"Raw data shape: {raw_df.shape}")

    print("\nCreating processed dataset...")
    processed_df = create_model_dataset(
        raw_df,
        lookahead_days=LOOKAHEAD_DAYS,
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD
    )

    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nProcessed data saved successfully.")
    print(f"Saved to: {PROCESSED_DATA_PATH}")
    print(f"Processed data shape: {processed_df.shape}")

    print("\nSignal distribution:")
    print(processed_df["Signal"].value_counts())

    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test, train_df, test_df = split_time_series_data(
        processed_df,
        test_size_ratio=TEST_SIZE_RATIO
    )

    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")

    print("\nTraining models...")
    models = train_models(X_train, y_train)

    print("\nEvaluating models...")
    results_df = evaluate_models(models, X_test, y_test)

    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nModel results saved successfully.")
    print(f"Saved to: {RESULTS_PATH}")
    print(results_df)


if __name__ == "__main__":
    main()
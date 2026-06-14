import os
import pandas as pd

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    RESULTS_PATH,
    BACKTEST_PATH,
    EQUITY_CURVE_PATH,
    INITIAL_CAPITAL,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    LOOKAHEAD_DAYS,
    TEST_SIZE_RATIO
)

from load_data import load_spx_data
from feature_engineering import create_model_dataset


def main():

    print("Loading raw data...")
    raw_df = load_spx_data(RAW_DATA_PATH)

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
    print(processed_df.head())
    print(f"Processed data shape: {processed_df.shape}")


if __name__ == "__main__":
    main()
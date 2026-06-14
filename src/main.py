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

def main():

    print("Loading raw data...")
    raw_df = load_spx_data(RAW_DATA_PATH)

    print(raw_df.head())

if __name__ == "__main__":
    main()
import pandas as pd

from config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
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
from train_models import (
    split_time_series_data,
    train_models,
    evaluate_models
)
from backtest import (
    convert_xgboost_predictions,
    run_backtest,
    calculate_backtest_metrics,
    plot_equity_curve
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

    print("\nRunning backtest using XGBoost...")

    xgboost_model = models["XGBoost"]
    xgboost_predictions_encoded = xgboost_model.predict(X_test)
    xgboost_predictions = convert_xgboost_predictions(xgboost_predictions_encoded)

    backtest_df = run_backtest(
        test_df=test_df,
        predictions=xgboost_predictions,
        initial_capital=INITIAL_CAPITAL
    )

    backtest_df.to_csv(BACKTEST_PATH, index=False)

    backtest_metrics = calculate_backtest_metrics(backtest_df)
    backtest_metrics_df = pd.DataFrame([backtest_metrics])

    print("\nBacktest results:")
    print(backtest_metrics_df)

    print("\nBacktest data saved successfully.")
    print(f"Saved to: {BACKTEST_PATH}")

    plot_equity_curve(backtest_df, EQUITY_CURVE_PATH)

    print("\nEquity curve saved successfully.")
    print(f"Saved to: {EQUITY_CURVE_PATH}")


if __name__ == "__main__":
    main()
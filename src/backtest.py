import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert_xgboost_predictions(predictions):
    mapping = {
        0: -1,
        1: 0,
        2: 1
    }

    return pd.Series(predictions).map(mapping).values


def run_backtest(test_df: pd.DataFrame, predictions, initial_capital: float) -> pd.DataFrame:
    backtest_df = test_df.copy()
    backtest_df["Predicted_Signal"] = predictions

    backtest_df["Daily_Return"] = backtest_df["Close"].pct_change()

    position = 0
    positions = []

    for signal in backtest_df["Predicted_Signal"]:
        if signal == 1:
            position = 1
        elif signal == -1:
            position = 0

        positions.append(position)

    backtest_df["Position"] = positions

    backtest_df["Strategy_Return"] = (
        backtest_df["Position"].shift(1) * backtest_df["Daily_Return"]
    )

    backtest_df["Strategy_Return"] = backtest_df["Strategy_Return"].fillna(0)
    backtest_df["Buy_Hold_Return"] = backtest_df["Daily_Return"].fillna(0)

    backtest_df["Strategy_Equity"] = initial_capital * (
        1 + backtest_df["Strategy_Return"]
    ).cumprod()

    backtest_df["Buy_Hold_Equity"] = initial_capital * (
        1 + backtest_df["Buy_Hold_Return"]
    ).cumprod()

    return backtest_df


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1

    return drawdown.min()


def calculate_sharpe_ratio(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0

    return np.sqrt(252) * returns.mean() / returns.std()


def calculate_backtest_metrics(backtest_df: pd.DataFrame) -> dict:
    strategy_final_value = backtest_df["Strategy_Equity"].iloc[-1]
    buy_hold_final_value = backtest_df["Buy_Hold_Equity"].iloc[-1]

    strategy_total_return = (
        strategy_final_value / backtest_df["Strategy_Equity"].iloc[0] - 1
    )

    buy_hold_total_return = (
        buy_hold_final_value / backtest_df["Buy_Hold_Equity"].iloc[0] - 1
    )

    strategy_sharpe = calculate_sharpe_ratio(backtest_df["Strategy_Return"])
    buy_hold_sharpe = calculate_sharpe_ratio(backtest_df["Buy_Hold_Return"])

    strategy_max_drawdown = calculate_max_drawdown(backtest_df["Strategy_Equity"])
    buy_hold_max_drawdown = calculate_max_drawdown(backtest_df["Buy_Hold_Equity"])

    number_of_trades = backtest_df["Position"].diff().abs().sum()

    metrics = {
        "Strategy_Final_Value": strategy_final_value,
        "Buy_Hold_Final_Value": buy_hold_final_value,
        "Strategy_Total_Return": strategy_total_return,
        "Buy_Hold_Total_Return": buy_hold_total_return,
        "Strategy_Sharpe": strategy_sharpe,
        "Buy_Hold_Sharpe": buy_hold_sharpe,
        "Strategy_Max_Drawdown": strategy_max_drawdown,
        "Buy_Hold_Max_Drawdown": buy_hold_max_drawdown,
        "Number_of_Trades": number_of_trades
    }

    return metrics


def plot_equity_curve(backtest_df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(12, 6))

    plt.plot(
        backtest_df["Date"],
        backtest_df["Strategy_Equity"],
        label="ML Strategy"
    )

    plt.plot(
        backtest_df["Date"],
        backtest_df["Buy_Hold_Equity"],
        label="Buy and Hold"
    )

    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("ML Strategy vs Buy and Hold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
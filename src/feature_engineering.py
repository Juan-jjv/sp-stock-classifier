import numpy as np
import pandas as pd


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["Return_20d"] = df["Close"].pct_change(20)

    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()

    df["Close_to_MA10"] = df["Close"] / df["MA_10"] - 1
    df["Close_to_MA50"] = df["Close"] / df["MA_50"] - 1
    df["Close_to_MA200"] = df["Close"] / df["MA_200"] - 1

    df["Volatility_10"] = df["Return_1d"].rolling(window=10).std()
    df["Volatility_20"] = df["Return_1d"].rolling(window=20).std()
    df["Volatility_50"] = df["Return_1d"].rolling(window=50).std()

    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_Return"] = (df["Close"] - df["Open"]) / df["Open"]

    df["Volume_Change"] = df["Volume"].pct_change(1)
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_to_MA20"] = df["Volume"] / df["Volume_MA_20"] - 1

    df["RSI_14"] = calculate_rsi(df["Close"])

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    return df


def add_target(
    df: pd.DataFrame,
    lookahead_days: int,
    buy_threshold: float,
    sell_threshold: float
) -> pd.DataFrame:
    df = df.copy()

    df["Future_Close"] = df["Close"].shift(-lookahead_days)
    df["Future_Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]

    conditions = [
        df["Future_Return"] > buy_threshold,
        df["Future_Return"] < sell_threshold
    ]

    choices = [1, -1]

    df["Signal"] = np.select(conditions, choices, default=0)

    return df


def create_model_dataset(
    df: pd.DataFrame,
    lookahead_days: int,
    buy_threshold: float,
    sell_threshold: float
) -> pd.DataFrame:
    df = add_features(df)
    df = add_target(df, lookahead_days, buy_threshold, sell_threshold)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df
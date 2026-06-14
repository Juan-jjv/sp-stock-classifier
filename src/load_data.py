import pandas as pd


def load_spx_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df = df.drop_duplicates(subset=["Date"])

    required_columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    return df
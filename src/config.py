from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "SPX.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "spx_features.csv"

MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"

RESULTS_PATH = BASE_DIR / "outputs" / "model_results.csv"
BACKTEST_PATH = BASE_DIR / "outputs" / "backtest_results.csv"
EQUITY_CURVE_PATH = BASE_DIR / "outputs" / "equity_curve.png"

INITIAL_CAPITAL = 10000

BUY_THRESHOLD = 0.01
SELL_THRESHOLD = -0.01

LOOKAHEAD_DAYS = 5
TEST_SIZE_RATIO = 0.2
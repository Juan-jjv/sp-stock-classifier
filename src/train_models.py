import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier


def get_feature_columns() -> list:
    return [
        "Return_1d",
        "Return_5d",
        "Return_10d",
        "Return_20d",
        "Close_to_MA10",
        "Close_to_MA50",
        "Close_to_MA200",
        "Volatility_10",
        "Volatility_20",
        "Volatility_50",
        "High_Low_Range",
        "Close_Open_Return",
        "Volume_Change",
        "Volume_to_MA20",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist"
    ]


def split_time_series_data(df: pd.DataFrame, test_size_ratio: float):
    split_index = int(len(df) * (1 - test_size_ratio))

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    feature_columns = get_feature_columns()

    X_train = train_df[feature_columns]
    y_train = train_df["Signal"]

    X_test = test_df[feature_columns]
    y_test = test_df["Signal"]

    return X_train, X_test, y_train, y_test, train_df, test_df


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42
        )
    }

    trained_models = {}

    y_train_xgb = y_train.map({-1: 0, 0: 1, 1: 2})

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        if model_name == "XGBoost":
            model.fit(X_train, y_train_xgb)
        else:
            model.fit(X_train, y_train)

        trained_models[model_name] = model

    return trained_models


def evaluate_models(models, X_test, y_test) -> pd.DataFrame:
    results = []

    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")

        if model_name == "XGBoost":
            encoded_predictions = model.predict(X_test)
            y_pred = pd.Series(encoded_predictions).map({0: -1, 1: 0, 2: 1})
        else:
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision_Macro": precision,
            "Recall_Macro": recall,
            "F1_Macro": f1
        })

        print(classification_report(y_test, y_pred, zero_division=0))

    return pd.DataFrame(results)
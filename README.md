# S&P 500 Trading Signal Prediction

The S&P 500 Trading Signal Prediction project uses historical S&P 500 OHLCV market data to predict Buy, Hold, and Sell trading signals based on future price movement. The project uses Python, pandas, scikit-learn, and XGBoost to build multiclass classification models and evaluate their performance using accuracy, precision, recall, and F1-score.

## Features

### Historical Market Data Processing

The project loads historical S&P 500 daily market data from `SPX.csv`. The dataset includes Date, Open, High, Low, Close, Adjusted Close, and Volume columns.

### Feature Engineering

The project creates technical indicators and time-series features from the raw market data, including:

* Daily returns
* 5-day, 10-day, and 20-day returns
* Moving averages
* Rolling volatility
* RSI
* MACD
* High-low price range
* Close-open return
* Volume change

### Processed Dataset

After loading the raw dataset and applying feature engineering, the processed dataset is saved to:

```text
data/processed/spx_features.csv
```

The processed dataset contains the original market data, engineered technical indicators, future returns, and the final Buy, Hold, and Sell target signal used for model training.

### Multiclass Classification

The project trains machine learning models to predict three trading signal classes:

```text
-1: Sell
 0: Hold
 1: Buy
```

The target signal is created using future returns. If the future return is above the buy threshold, the signal is labeled as Buy. If the future return is below the sell threshold, the signal is labeled as Sell. Otherwise, it is labeled as Hold.

### Models Used

The project trains and compares the following models:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost

### Model Evaluation

The models are evaluated using classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score

The model comparison results are saved to:

```text
outputs/model_results.csv
```

## Installation

### Prerequisites

Python 3.x

### Steps

Clone the repository:

```bash
git clone https://github.com/Juan-jjv/sp-stock-classifier.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the main pipeline:

```bash
python src/main.py
```

This will load the raw S&P 500 dataset, create engineered features, generate Buy, Hold, and Sell target signals, train the machine learning models, and save the model evaluation results.

## Project Structure

```text
SP_Stock_Classifier/
│
├── data/
│   ├── raw/
│   │   └── SPX.csv
│   └── processed/
│       └── spx_features.csv
│
├── outputs/
│   └── model_results.csv
│
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── main.py
│
├── README.md
└── requirements.txt
```

## Current Results

The initial models showed moderate classification performance. XGBoost and Gradient Boosting achieved the highest accuracy, while Logistic Regression produced the strongest macro F1-score.

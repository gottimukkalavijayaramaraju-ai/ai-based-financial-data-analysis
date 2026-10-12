# AI-Based Financial Data Analysis

Loads stock data, builds technical indicators, trains ML models (Random Forest, Linear Regression, Isolation Forest), backtests a simple strategy and saves charts.

## Setup
    pip install -r requirements.txt

## Run
    python financial_analysis.py
    python financial_analysis.py --ticker TSLA --period 3y
    python financial_analysis.py --csv mydata.csv   # needs Date, Close columns

Results are saved in the `output/` folder.

Disclaimer: educational project only - not financial advice.

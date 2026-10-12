"""
AI-Based Financial Data Analysis  (base project)
------------------------------------------------
Pipeline:
  1. Load data        -> real stock prices (yfinance) or CSV, with synthetic fallback
  2. Feature building -> returns, moving averages, RSI, MACD, volatility
  3. AI models        -> Random Forest (next-day direction), Linear Regression (next-day price),
                         Isolation Forest (anomaly detection)
  4. Evaluation       -> accuracy, RMSE, backtest of a simple strategy
  5. Visualization    -> saved charts in ./output

Usage:
  python financial_analysis.py                      # default AAPL, 5 years
  python financial_analysis.py --ticker TSLA --period 3y
  python financial_analysis.py --csv mydata.csv     # needs Date, Close columns
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- 1. DATA
def load_data(ticker="AAPL", period="5y", csv_path=None):
    """Load price data from CSV, Yahoo Finance, or fall back to synthetic data."""
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date").sort_index()
        print(f"[data] Loaded {len(df)} rows from {csv_path}")
        return df

    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise ValueError("empty download")
        print(f"[data] Downloaded {len(df)} rows for {ticker} from Yahoo Finance")
        return df
    except Exception as e:
        print(f"[data] Could not download data ({e}). Using synthetic data instead.")
        return make_synthetic_data()


def make_synthetic_data(n=1250, seed=42):
    """Geometric Brownian motion price series for offline testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    returns[rng.choice(n, 8, replace=False)] *= 5  # inject shocks
    close = 100 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.003, n)),
        "High": close * (1 + abs(rng.normal(0, 0.006, n))),
        "Low": close * (1 - abs(rng.normal(0, 0.006, n))),
        "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)


# ------------------------------------------------------- 2. FEATURES
def add_features(df):
    df = df.copy()
    c = df["Close"]
    df["Return"] = c.pct_change()
    df["SMA_10"] = c.rolling(10).mean()
    df["SMA_50"] = c.rolling(50).mean()
    df["Volatility_10"] = df["Return"].rolling(10).std()

    # RSI (14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)

    # MACD
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # Lagged returns
    for lag in (1, 2, 3, 5):
        df[f"Ret_lag{lag}"] = df["Return"].shift(lag)

    df["Price_to_SMA50"] = c / df["SMA_50"]

    # Targets
    df["Target_Price"] = c.shift(-1)
    df["Target_Up"] = (c.shift(-1) > c).astype(int)
    return df


FEATURES = ["Return", "SMA_10", "SMA_50", "Volatility_10", "RSI", "MACD",
            "MACD_signal", "Ret_lag1", "Ret_lag2", "Ret_lag3", "Ret_lag5",
            "Price_to_SMA50"]


# ------------------------------------------------------- 3. AI MODELS
def train_models(df, test_size=0.2):
    data = df.dropna(subset=FEATURES + ["Target_Price"]).copy()
    split = int(len(data) * (1 - test_size))  # time-based split, no shuffling
    train, test = data.iloc[:split], data.iloc[split:]

    scaler = StandardScaler().fit(train[FEATURES])
    X_tr, X_te = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])

    # Direction classifier
    clf = RandomForestClassifier(n_estimators=300, max_depth=6,
                                 min_samples_leaf=5, random_state=42)
    clf.fit(X_tr, train["Target_Up"])
    pred_dir = clf.predict(X_te)
    print("\n=== Direction Prediction (Random Forest) ===")
    print(f"Accuracy: {accuracy_score(test['Target_Up'], pred_dir):.3f}")
    print(classification_report(test["Target_Up"], pred_dir, zero_division=0))

    # Price regressor
    reg = LinearRegression().fit(X_tr, train["Target_Price"])
    pred_price = reg.predict(X_te)
    rmse = np.sqrt(mean_squared_error(test["Target_Price"], pred_price))
    print("=== Price Prediction (Linear Regression) ===")
    print(f"RMSE: {rmse:.3f}")

    # Anomaly detection on returns & volatility
    iso = IsolationForest(contamination=0.02, random_state=42)
    data["Anomaly"] = iso.fit_predict(data[["Return", "Volatility_10"]]) == -1
    print(f"\n=== Anomaly Detection ===\nFlagged {data['Anomaly'].sum()} unusual days")

    test = test.copy()
    test["Pred_Up"] = pred_dir
    test["Pred_Price"] = pred_price

    importance = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
    return test, data, importance


# ------------------------------------------------------- 4. BACKTEST
def backtest(test):
    """Go long when the model predicts 'up', otherwise stay in cash."""
    strat_ret = test["Pred_Up"].values * test["Return"].shift(-1).fillna(0).values
    test = test.copy()
    test["Strategy"] = (1 + strat_ret).cumprod()
    test["BuyHold"] = (1 + test["Return"].shift(-1).fillna(0)).cumprod()
    sharpe = np.sqrt(252) * strat_ret.mean() / (strat_ret.std() + 1e-9)
    print("\n=== Backtest ===")
    print(f"Strategy return : {test['Strategy'].iloc[-1] - 1:+.2%}")
    print(f"Buy & hold      : {test['BuyHold'].iloc[-1] - 1:+.2%}")
    print(f"Sharpe (strategy): {sharpe:.2f}")
    return test


# ------------------------------------------------------- 5. CHARTS
def make_charts(df, data, test, importance):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax[0].plot(df.index, df["Close"], label="Close")
    ax[0].plot(df.index, df["SMA_10"], label="SMA 10", alpha=.7)
    ax[0].plot(df.index, df["SMA_50"], label="SMA 50", alpha=.7)
    an = data[data["Anomaly"]]
    ax[0].scatter(an.index, an["Close"], color="red", s=20, label="Anomaly", zorder=5)
    ax[0].set_title("Price, Moving Averages & Anomalies"); ax[0].legend()
    ax[1].plot(df.index, df["RSI"], color="purple"); ax[1].axhline(70, ls="--", c="r")
    ax[1].axhline(30, ls="--", c="g"); ax[1].set_title("RSI (14)")
    ax[2].plot(df.index, df["MACD"], label="MACD")
    ax[2].plot(df.index, df["MACD_signal"], label="Signal"); ax[2].legend()
    ax[2].set_title("MACD")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/technical_analysis.png", dpi=120); plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(test.index, test["Target_Price"], label="Actual")
    plt.plot(test.index, test["Pred_Price"], label="Predicted", alpha=.8)
    plt.title("Next-Day Price: Actual vs Predicted"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/price_prediction.png", dpi=120); plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(test.index, test["Strategy"], label="AI Strategy")
    plt.plot(test.index, test["BuyHold"], label="Buy & Hold")
    plt.title("Backtest"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/backtest.png", dpi=120); plt.close()

    plt.figure(figsize=(7, 5)); importance.plot(kind="barh")
    plt.title("Feature Importance"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=120); plt.close()
    print(f"\n[charts] Saved to ./{OUT_DIR}/")


# ------------------------------------------------------- MAIN
def main():
    p = argparse.ArgumentParser(description="AI-Based Financial Data Analysis")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--period", default="5y")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    raw = load_data(args.ticker, args.period, args.csv)
    df = add_features(raw)
    test, data, importance = train_models(df)
    test = backtest(test)
    make_charts(df, data, test, importance)
    df.to_csv(f"{OUT_DIR}/processed_data.csv")
    print("[done] Processed data saved to output/processed_data.csv")
    print("\nDisclaimer: educational project only - not financial advice.")


if __name__ == "__main__":
    main()

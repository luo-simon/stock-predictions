import argparse
import os
import talib
from src.misc import load_csv_to_df
import numpy as np

"""Assess the raw data. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Create visualisation routines to assess the data. Ensure that date formats are correct and correctly timezoned."""


def preprocess(raw_path, processed_path):
    for file in os.listdir(raw_path):
        df = load_csv_to_df(os.path.join(raw_path, file))
        df = generate_features(df)

        # Remove first 50 rows, as will be null for e.g. SMA_50 and final row, as no label
        df = df.iloc[50:-1]

        # Ensure no missing values
        assert df.isna().any(axis=1).sum() == 0

        df.to_csv(os.path.join(processed_path, file), index=True, header=True)


def add_technical_indicators(df):
    """
    Adds selected technical indicators to the input DataFrame using TA-Lib.

    Parameters:
    df (pd.DataFrame): Stock price data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].

    Returns:
    pd.DataFrame: The original DataFrame augmented with technical indicators.
    """
    # Overlap Studies
    ## Bollinger Bands
    df["upper_band"], df["middle_band"], df["lower_band"] = talib.BBANDS(
        df["Close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    ## Simple Moving Average
    df["SMA_3"] = talib.SMA(df["Close"], timeperiod=3)
    df["SMA_5"] = talib.SMA(df["Close"], timeperiod=5)
    df["SMA_10"] = talib.SMA(df["Close"], timeperiod=10)
    df["SMA_20"] = talib.SMA(df["Close"], timeperiod=20)
    df["SMA_50"] = talib.SMA(df["Close"], timeperiod=50)
    ## Exponential Moving Average
    df["EMA_3"] = talib.EMA(df["Close"], timeperiod=3)
    df["EMA_5"] = talib.EMA(df["Close"], timeperiod=5)
    df["EMA_10"] = talib.EMA(df["Close"], timeperiod=10)
    df["EMA_20"] = talib.EMA(df["Close"], timeperiod=20)
    df["EMA_50"] = talib.EMA(df["Close"], timeperiod=50)

    # Momentum Indicators
    # Average Directional Movement Index
    df["ADX"] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
    # Aroon
    df["aroon_down"], df["aroon_up"] = talib.AROON(df["High"], df["Low"], timeperiod=14)
    # MACD (Moving Average Convergence Divergence)
    df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(
        df["Close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    # Relative Strength Index (RSI)
    df["RSI_14"] = talib.RSI(df["Close"], timeperiod=14)
    # Stochastic Oscillator
    df["slow_k"], df["slow_d"] = talib.STOCH(
        df["High"],
        df["Low"],
        df["Close"],
        fastk_period=5,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    # Williams %R
    df["williams_r"] = talib.WILLR(df["High"], df["Low"], df["Close"], timeperiod=14)

    # Volume Indicators
    # Accumulation/Distribution Line
    df["AD"] = talib.AD(df["High"], df["Low"], df["Close"], df["Volume"])
    # On-Balance Volume
    df["OBV"] = talib.OBV(df["Close"], df["Volume"])

    # Volatility Indicators
    df["NATR"] = talib.NATR(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["TRANGE"] = talib.TRANGE(df["High"], df["Low"], df["Close"])

    return df


def generate_features(df):

    # Transformations
    df["log_open"] = np.log(df["Open"])
    df["log_high"] = np.log(df["High"])
    df["log_low"] = np.log(df["Low"])
    df["log_close"] = np.log(df["Close"])
    df["log_volume"] = np.log(df["Volume"])

    # Target
    df["Close Forecast"] = df["Close"].shift(-1)
    # df["log Close Forecast"] = np.log(df["Close Forecast"])

    # Close Price Lagged
    df["close_t-1"] = df["Close"].shift(1)  # Lag of 1 day
    df["close_t-2"] = df["Close"].shift(2)  # Lag of 2 days
    df["close_t-3"] = df["Close"].shift(3)  # Lag of 3 days
    df["close_t-4"] = df["Close"].shift(4)  # Lag of 4 days
    df["close_t-5"] = df["Close"].shift(5)  # Lag of 5 days

    # Pct returns
    df["pct_change"] = df["Close"].pct_change()

    # Log return
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Date-related ("categorical")
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"] = df.index.day
    df["weekofyear"] = df.index.isocalendar().week

    # Technical Indicators
    df = add_technical_indicators(df)

    # Macroeconomic Indicator (FFR)
    fed_funds = load_csv_to_df("data/external/FEDFUNDS.csv")["FEDFUNDS"].rename(
        "fed_funds_rate"
    )
    fed_funds = fed_funds.reindex(df.index, method="ffill")
    df = df.join(fed_funds, how="left")
    df["log_fed_funds_rate"] = np.log(df["fed_funds_rate"])

    # Market Indices # Todo: Data Leakage
    external_path = "data/external"
    for filename in os.listdir(external_path):
        file_path = os.path.join(external_path, filename)
        if os.path.isfile(file_path) and "^" in filename:
            col_name = filename.split(".")[0]
            index_series = load_csv_to_df(os.path.abspath(file_path))["Close"]
            index_series = index_series.rename(col_name)
            index_series = index_series.reindex(df.index, method="ffill")
            df = df.join(index_series, how="left").dropna()
            df[f"log_{col_name}"] = np.log(df[col_name])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("-r", "--raw-path", help="Raw data path", default="data/raw")
    parser.add_argument(
        "-p", "--processed-path", help="Processed data path", default="data/processed"
    )
    args = parser.parse_args()

    preprocess(args.raw_path, args.processed_path)

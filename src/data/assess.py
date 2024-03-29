import argparse
import os
import pandas as pd
import talib
from datetime import datetime, timedelta
from ..misc import load_csv_to_df

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


def generate_features(df):
    # Labels
    df["Close Forecast"] = df["Close"].shift(-1)

    # Close Price Lagged
    df['Close_T-1'] = df['Close'].shift(1)  # Lag of 1 day
    df['Close_T-2'] = df['Close'].shift(2)  # Lag of 2 days
    df['Close_T-5'] = df['Close'].shift(5)  # Lag of 5 days

    # Simple Moving Average
    df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50) 

    # Exponential Moving Average
    df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
    df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)

    # Relative Strength Index (RSI)
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)

    # Bollinger Bands
    df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # MACD (Moving Average Convergence Divergence)
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # On-Balance Volume
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    # Average Directional Movement Index
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Accumulation/Distribution Line
    df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])

    # Aroon
    df['Aroon_down'], df['Aroon_up'] = talib.AROON(df['High'], df['Low'], timeperiod=14)

    # Stochastic Oscillator
    df['SlowK'], df['SlowD'] = talib.STOCH(df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # Williams %R
    df['Williams %R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)

    # Pct returns
    df["pct_change"] = df["Close"].pct_change()

    # Date-related
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    # Index-prices
    sp_close = load_csv_to_df("/Users/simon/Documents/II/Dissertation/data/external/^SPX.csv")["Close"]
    df["S&P Close"] = sp_close

    

    return df

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("-r", "--raw-path", help="Raw data path", required=True)
    parser.add_argument("-p", "--processed-path", help="Processed data path", required=True)
    args = parser.parse_args()

    preprocess(args.raw_path, args.processed_path)
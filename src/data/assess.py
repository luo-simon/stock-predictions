import argparse
import os
import pandas as pd
import talib
from datetime import datetime, timedelta

"""Assess the raw data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Create visualisation routines to assess the data. Ensure that date formats are correct and correctly timezoned."""

def load_raw_data(path):
    """Load raw dataset located at path"""
    df = pd.read_csv(path)
    df = generate_features(df)
    df.index = df.index.tz_localize(None)
    return

def preprocess(raw_path, preprocessed_path):
    for file in os.listdir(raw_path):
        print(file)

def generate_features(df):
    """Generate features from data DataFrame."""

    # Labels
    df["Close Forecast"] = df["Close"].shift(-1)

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
    df["S&P Close"] = access.data("^spx", start=df.index[0], end=df.index[-1])["Close"]

    return df

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""

    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("-r", "--raw-path", help="Raw data path", required=True)
    parser.add_argument("-p", "--preprocessed-path", help="Preprocessed data path", required=True)
    args = parser.parse_args()

    preprocess(args.raw_path, args.preprocessed_path)
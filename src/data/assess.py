import argparse
import os
import talib
from src.misc import load_csv_to_df
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

"""Assess the raw data. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Create visualisation routines to assess the data. Ensure that date formats are correct and correctly timezoned."""


def preprocess(raw_path, processed_path):
    for file in os.listdir(raw_path):
        df = load_csv_to_df(os.path.join(raw_path, file))[["Open", "High", "Low", "Close", "Volume"]]

        df = df["1999-01-01":]
        
        # Assert no missing values before feature generation
        assert df.isna().any(axis=1).sum() == 0, f"Missing values found {file}"
        zeros = df[(df==0).any(axis=1)]
        if zeros.shape[0] > 0:
            print(f"Zero values found {file} {zeros}")

        # Generate features
        df = generate_features(df)
        # Assert no missing values after feature generation
        assert df.isna().any(axis=1).sum() == 0, f"Missing values found {file}"
                

        suffixes = {
            "" : "Close (log return)",
            "_open" : "Open (log return)", 
            "_high": "High (log return)",
            "_low" : "Low (log return)",
            "_volume": "Volume (log return)"
        }
       
        # Treat outliers
        print(file)
        fig, axs = plt.subplots(1, 5, figsize=(20, 3), sharey=True) 
        for i, (k, v) in enumerate(suffixes.items()):
            sns.histplot(df[f'log_return{k}'], kde=False, ax=axs[i], stat="count")
            # axs[i].set_title(f'Distribution of {v}')
            axs[i].set_xlabel(f'{v}')
            axs[i].legend()
        fig.tight_layout()
        
        df["log_return"] = treat_outliers(df["log_return"])
        df["log_return_open"] = treat_outliers(df["log_return_open"])
        df["log_return_high"] = treat_outliers(df["log_return_high"])
        df["log_return_low"] = treat_outliers(df["log_return_low"])
        df["log_return_volume"] = treat_outliers(df["log_return_volume"])

        fig, axs = plt.subplots(1, 5, figsize=(20, 3), sharey=True) 
        for i, (k, v) in enumerate(suffixes.items()):
            col = df[f'log_return{k}']
            sns.histplot(col, kde=False, ax=axs[i], stat="count")
            # axs[i].set_title(f'Distribution of {v}')
            axs[i].set_xlabel(f'{v}')
            # x = np.linspace(min(col), max(col), 100)
            # p = stats.norm.pdf(x, col.mean(), col.std()-0.003)
            # axs[i].plot(x, p, 'k', linewidth=1, label=f'Normal distribution', linestyle="--")
        fig.tight_layout()
        # plt.show() 

        df.to_csv(os.path.join(processed_path, file), index=True, header=True)


def add_technical_indicators(df):
    """
    Adds selected technical indicators to the input DataFrame using TA-Lib.

    Parameters:
    df (pd.DataFrame): Stock price data with columns ['Open', 'High', 'Low', 'Close', 'Volume'].

    Returns:
    pd.DataFrame: The original DataFrame augmented with technical indicators.
    """

    ## Moving Averages
    sma = talib.SMA(df["Close"], timeperiod=10)
    wma = talib.WMA(df["Close"], timeperiod=10)
    ema = talib.EMA(df["Close"], timeperiod=10)
    dema = talib.DEMA(df["Close"], timeperiod=10)
    tema = talib.TEMA(df["Close"], timeperiod=10)
    df["sma"] = np.where(df["Close"] > sma, 1, -1)
    df["wma"] = np.where(df["Close"] > wma, 1, -1)
    df["ema"] = np.where(df["Close"] > ema, 1, -1)
    df["dema"] = np.where(df["Close"] > dema, 1, -1)
    df["tema"] = np.where(df["Close"] > tema, 1, -1)

    # Other technical indicators
    aroon_down, aroon_up = talib.AROON(df["High"], df["Low"], timeperiod=14)    # Aroon
    rsi = talib.RSI(df["Close"], timeperiod=14)                                 # Relative Strength Index (RSI)
    willr = talib.WILLR(df["High"], df["Low"], df["Close"], timeperiod=14)      # Williams R
    cci = talib.CCI(df["High"], df["Low"], df["Close"], timeperiod=14)          # CCI
    ad = talib.AD(df["High"], df["Low"], df["Close"], df["Volume"])             # Acculation/Distribution Line
    mom = talib.MOM(df["Close"], timeperiod=10)                                 # Momentum
    slowk, slowd = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0) # Stochastic K, D
    macd, macdsignal, macdhist = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9) # MACD

    df["aroon"] = np.where(aroon_up > aroon_down, 1, -1)
    df["rsi"] = np.where(rsi > 70, 1, np.where(rsi > rsi.shift(1), 1, -1))
    df["willr"] = np.where(willr > willr.shift(1), 1, -1)
    df["cci"] = np.where(cci > cci.shift(1), 1, -1)
    df["ad"] = np.where(ad > ad.shift(1), 1, -1)
    df["mom"] = np.where(mom > 0, 1, -1)
    df["slowk"] = np.where(slowk > slowk.shift(1), 1, -1)
    df["slowd"] = np.where(slowd > slowd.shift(1), 1, -1)
    df["macd"] = np.where(macd > macd.shift(1), 1, -1)
    
    return df.iloc[30:]

def treat_outliers(col):
    mean = col.mean()
    std = col.std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # Q1 = col.quantile(0.25)
    # Q3 = col.quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    return col.clip(lower=lower_bound, upper=upper_bound)

def generate_features(df):

    # Transformations
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return_open"] = np.log(df["Open"] / df["Open"].shift(1))
    df["log_return_high"] = np.log(df["High"] / df["High"].shift(1))
    df["log_return_low"] = np.log(df["Low"] / df["Low"].shift(1))
    df["log_return_volume"] = np.log(df["Volume"] / df["Volume"].shift(1))

    df["log_return_forecast"] = df["log_return"].shift(-1)

    # Technical Indicators
    df = add_technical_indicators(df)

    # Macroeconomic Indicator (FFR)
    fed_funds = load_csv_to_df("data/external/FEDFUNDS.csv")["FEDFUNDS"].rename(
        "fed_funds_rate"
    )
    fed_funds = fed_funds.reindex(df.index, method="ffill")
    df = df.join(fed_funds, how="left")

    # Market Indices # Todo: Data Leakage
    external_path = "data/external"
    for filename in os.listdir(external_path):
        file_path = os.path.join(external_path, filename)
        if os.path.isfile(file_path) and "^" in filename:
            col_name = filename.split(".")[0]
            index_series = load_csv_to_df(os.path.abspath(file_path))["Close"]
            index_series = index_series.rename(col_name)
            index_series = index_series.reindex(df.index, method="ffill")
            index_series = np.log(index_series/index_series.shift(1)) # Log return
            df = df.join(index_series, how="left").dropna()

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("-r", "--raw-path", help="Raw data path", default="data/raw")
    parser.add_argument(
        "-p", "--processed-path", help="Processed data path", default="data/processed"
    )
    args = parser.parse_args()

    preprocess(args.raw_path, args.processed_path)

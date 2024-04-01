"""
Preprocess data to produce suitable inputs to model
i.e. correct shape, datatype and normalised/scaled as appropriate
"""
from src.misc import load_processed_dataset


def load_data():
    """
    Returns X and y
    """
    df = load_processed_dataset("aapl", "2018-01-01", "2023-01-1")
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'SMA_10',
       'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50', 'RSI_14',
       'upper_band', 'middle_band', 'lower_band', 'macd', 'macdsignal',
       'macdhist', 'OBV', 'ADX', 'AD', 'Aroon_down', 'Aroon_up', 'SlowK',
       'SlowD', 'Williams %R', 'pct_change', 'dayofweek', 'quarter', 'month',
       'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'S&P Close']
    X = df[features]
    y = df["Close Forecast"]
    
    return X, y
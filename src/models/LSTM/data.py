"""
Preprocess data to produce suitable inputs to model
i.e. correct shape, datatype and normalised/scaled as appropriate
"""

from src.misc import load_processed_dataset
import pandas as pd


def load_data(features, ticker="aapl"):
    """
    Returns X and y
    """
    # Feature selection
    df = load_processed_dataset(ticker, "2018-01-01", "2023-01-1")

    X = df[features]
    y = pd.DataFrame(df["Close Forecast"])

    return X, y

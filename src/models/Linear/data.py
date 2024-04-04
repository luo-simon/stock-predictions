"""
Preprocess data to produce suitable inputs to model
i.e. correct shape, datatype and normalised/scaled as appropriate
"""

from src.misc import load_processed_dataset


def load_data(ticker="aapl"):
    """
    Returns X and y
    """
    df = load_processed_dataset(ticker, "2018-01-01", "2023-01-1")

    X = df.drop("Close Forecast", axis=1)
    y = df["Close Forecast"]

    return X, y

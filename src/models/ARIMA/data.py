"""
Preprocess data to produce suitable inputs to model
i.e. correct shape, datatype and normalised/scaled as appropriate
"""
from src.misc import load_processed_dataset


def load_data():
    """
    Returns X and y
    """
    df = load_processed_dataset("aapl", "2019-01-01", "2023-01-1")
    X = df["Close"]
    y = df["Close Forecast"]
    
    return X, y
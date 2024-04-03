"""
Preprocess data to produce suitable inputs to model
i.e. correct shape, datatype and normalised/scaled as appropriate
"""

from src.misc import load_processed_dataset, create_sequence
import pandas as pd


def load_data(features, sequence_len):
    """
    Returns X and y
    """
    # Feature selection
    df = load_processed_dataset("aapl", "2018-01-01", "2023-01-1")
    X = df[features].values
    y = pd.DataFrame(df["Close Forecast"]).values

    # Sequencing
    X = create_sequence(X, sequence_len)
    y = create_sequence(y, sequence_len)
    return X, y

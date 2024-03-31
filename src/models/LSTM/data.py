"""
Preprocess data to produce suitable inputs to model
i.e. correct shape, datatype and normalised/scaled as appropriate
"""

import numpy as np
from src.misc import load_processed_dataset
import torch


def load_data(features, sequence_len):
    """
    Returns X and y
    """
    # Feature selection
    df = load_processed_dataset("aapl", "2018-01-01", "2023-01-1")
    
    
    X = df[features]
    y = df["Close Forecast"]
    
    return X, y

def create_sequences(Xs, ys, sequence_length):
    """
    Given a numpy array, create sequences of a fixed length, where
    each sequence will be used to predict the closing price of the next day.
    """
    X = []
    y = []
    for i in range(sequence_length, len(Xs)+1):
        X.append(Xs[i-sequence_length:i])
        y.append(ys[i-1])
    return np.array(X), np.array(y)
"""
Preprocess data to produce suitable inputs to model
i.e. correct shape, datatype and normalised/scaled as appropriate
"""

import numpy as np
from src.misc import load_processed_dataset
import torch
import pandas as pd


def load_data(features):
    """
    Returns X and y
    """
    # Feature selection
    df = load_processed_dataset("aapl", "2018-01-01", "2023-01-1")

    X = df[features]
    y = pd.DataFrame(df["Close Forecast"])

    return X, y

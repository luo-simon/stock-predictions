"""
Preprocess data to produce suitable inputs to model
i.e. correct shape, datatype and normalised/scaled as appropriate
"""

import numpy as np
import misc
from sklearn.preprocessing import StandardScaler
import torch


def load_data(features, sequence_len):
    """
    Returns X and y
    """
    # Feature selection
    df = misc.load_processed_dataset("aapl", "2018-01-01", "2023-01-1")
    
    
    X = df[features]
    y = df["Close Forecast"]
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    y_scaled = scaler.fit_transform(y.values.reshape(-1,1))

    # Sequencing
    X_sequenced, y_sequenced = create_sequences(X_scaled, y_scaled, sequence_len)

    # Convert to tensors
    X_tensor = torch.tensor(X_sequenced.astype(np.float32))
    y_tensor = torch.tensor(y_sequenced.astype(np.float32))

    return X_tensor, y_tensor

def denorm(array):
    scaler = StandardScaler()
    scaler.inverse_transform(array.reshape(-1,1)).flatten()

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
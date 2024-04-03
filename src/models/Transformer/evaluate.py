import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from src.misc import (
    split_data,
    evaluate,
    plot,
    create_sequences,
    load_model_from_run_id,
    load_pytorch_model_from_latest_run,
    load_model_from_experiment,
)
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.models.Transformer.data import load_data



def eval(features, sequence_len, run_id):
    # Load data
    X, y = load_data(features=features, sequence_len=sequence_len)

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)


    # Normalise

    #    (n_samples, sequence_len, n_features)
    # -> (n_samples * sequence_len, n_features)
    X_train = X_train.reshape(-1, X.shape[2])
    X_test = X_test.reshape(-1, X.shape[2])
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    X_scaler = StandardScaler()
    X_scaler = X_scaler.fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_scaler = y_scaler.fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    X_train = X_train.reshape(-1, sequence_len, len(features))
    X_test = X_test.reshape(-1, sequence_len, len(features))
    y_train = y_train.reshape(-1, sequence_len, 1)
    y_test = y_test.reshape(-1, sequence_len, 1)

    # DataLoaders
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = load_model_from_run_id(run_id)

    # Evaluate
    actuals, predictions = [], []
    model.eval()
    with torch.no_grad(): 
        for X, y in test_loader:
                    preds = model(X)
                    preds = preds[:, -1:, :].squeeze()
                    y = y[:, -1:, :].squeeze()
                    actuals.extend(y)
                    predictions.extend(preds)
    actuals = y_scaler.inverse_transform(np.array(actuals).reshape(-1,1))
    predictions = y_scaler.inverse_transform(np.array(predictions).reshape(-1,1))

    r2, mse, rmse, mae, mape = evaluate(predictions, actuals, verbose=True)

    # Plot
    plot(predictions, actuals)

    return predictions, actuals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM model")
    parser.add_argument("--config-file", "-c", type=str, default="configs/lstm.yaml")
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "log_open",
        "log_high",
        "log_low",
        "log_close",
        "log_volume",
        "close_t-1",
        "close_t-2",
        "close_t-3",
        "close_t-4",
        "close_t-5",
        "pct_change",
        "log_return",
        "dayofweek",
        "quarter",
        "month",
        "year",
        "dayofyear",
        "dayofmonth",
        "weekofyear",
        "upper_band",
        "middle_band",
        "lower_band",
        "SMA_3",
        "SMA_5",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_3",
        "EMA_5",
        "EMA_10",
        "EMA_20",
        "EMA_50",
        "ADX",
        "aroon_down",
        "aroon_up",
        "macd",
        "macdsignal",
        "macdhist",
        "RSI_14",
        "slow_k",
        "slow_d",
        "williams_r",
        "AD",
        "OBV",
        "NATR",
        "TRANGE",
        "fed_funds_rate",
        "log_fed_funds_rate",
        "^N225",
        "log_^N225",
        "^IXIC",
        "log_^IXIC",
        "^FTSE",
        "log_^FTSE",
        "^SPX",
        "log_^SPX",
        "^DJI",
        "log_^DJI",
    ]
    sequence_len = 5
    run_id = "9696617090e64aa9be89cf24aea4e6ca"

    eval(features, sequence_len, run_id)

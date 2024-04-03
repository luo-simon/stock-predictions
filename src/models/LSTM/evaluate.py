import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from src.models.LSTM.data import load_data
from src.misc import split_data, evaluate, plot, create_sequences, load_model_from_run_id, load_pytorch_model_from_latest_run, load_model_from_experiment
from sklearn.preprocessing import StandardScaler
import pandas as pd


def eval(features, sequence_len, run_id):
    # Load test data
    # Load data
    X, y = load_data(features=features)

    # Split
    X_train, _, X_test = split_data(X, verbose=False)
    y_train, _, y_test = split_data(y, verbose=False)

    # Normalisation
    in_scaler = StandardScaler()
    out_scaler = StandardScaler()
    _ = in_scaler.fit_transform(X_train.values)
    X_test_norm = in_scaler.transform(X_test.values)
    _ = out_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_norm = out_scaler.transform(y_test.values.reshape(-1, 1))

    # Sequencing
    X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test_norm, sequence_len)

    # TensorDatasets and DataLoaders
    test_dataset = TensorDataset(
        torch.tensor(X_test_seq.astype(np.float32)),
        torch.tensor(y_test_seq.astype(np.float32)),
    )

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = load_model_from_run_id(run_id)
    
    # Predict:
    predictions = []
    actuals = []
    for Xs, ys in test_loader:
        output = model(Xs)
        predictions.extend(output.flatten().tolist())
        actuals.extend(ys.flatten().tolist())
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    predictions_rescaled = out_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).flatten()
    actuals_rescaled = out_scaler.inverse_transform(
        actuals.reshape(-1, 1)
    ).flatten()

    preds = pd.Series(predictions_rescaled, index=y_test[sequence_len - 1 :].index)
    obs = pd.Series(actuals_rescaled, index=y_test[sequence_len - 1 :].index)

    r2, mse, rmse, mae, mape = evaluate(preds, obs, verbose=True)

    # Plot
    plot(preds, obs)

    return preds, obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LSTM model")
    parser.add_argument("--config-file", "-c", type=str, default="configs/lstm.yaml")
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'log_open', 'log_high', 'log_low', 'log_close', 'log_volume', 'close_t-1', 'close_t-2', 'close_t-3', 'close_t-4', 'close_t-5', 'pct_change', 'log_return', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'upper_band', 'middle_band', 'lower_band', 'SMA_3', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_3', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'ADX', 'aroon_down', 'aroon_up', 'macd', 'macdsignal', 'macdhist', 'RSI_14', 'slow_k', 'slow_d', 'williams_r', 'AD', 'OBV', 'NATR', 'TRANGE', 'fed_funds_rate', 'log_fed_funds_rate', '^N225', 'log_^N225', '^IXIC', 'log_^IXIC', '^FTSE', 'log_^FTSE', '^SPX', 'log_^SPX', '^DJI', 'log_^DJI']
    sequence_len = 5
    run_id='9696617090e64aa9be89cf24aea4e6ca'

    eval(features, sequence_len, run_id)

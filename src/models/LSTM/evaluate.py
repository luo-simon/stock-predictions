import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from src.models.LSTM.data import load_data
from src.misc import split_data, evaluate, plot, create_sequences
from sklearn.preprocessing import StandardScaler
import pandas as pd


def eval(features, sequence_len, experiment_name):
    # Load test data
    # Load data
    X, y = load_data(features=features, sequence_len=sequence_len)
    
    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)
    
    # Normalisation
    in_scaler = StandardScaler()
    out_scaler = StandardScaler()
    X_train_norm = in_scaler.fit_transform(X_train.values)
    X_val_norm  = in_scaler.transform(X_val.values)
    X_test_norm  = in_scaler.transform(X_test.values)
    y_train_norm = out_scaler.fit_transform(y_train.values.reshape(-1,1))
    y_val_norm  = out_scaler.transform(y_val.values.reshape(-1,1))
    y_test_norm  = out_scaler.transform(y_test.values.reshape(-1,1))
    
    # Sequencing
    X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train_norm, sequence_len)
    X_val_seq, y_val_seq = create_sequences(X_val_norm, y_val_norm, sequence_len)
    X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test_norm, sequence_len)

    # TensorDatasets and DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train_seq.astype(np.float32)), torch.tensor(y_train_seq.astype(np.float32)))
    val_dataset = TensorDataset(torch.tensor(X_val_seq.astype(np.float32)), torch.tensor(y_val_seq.astype(np.float32)))
    test_dataset = TensorDataset(torch.tensor(X_test_seq.astype(np.float32)), torch.tensor(y_test_seq.astype(np.float32)))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) # Shuffle or no shuffle?
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load model:
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    # Search for runs in the experiment and get the latest run
    runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id], order_by=["start_time desc"])

    if not runs.empty:
        latest_run_id = runs.iloc[0]["run_id"]
        # Load the model from the latest run
        loaded_model = mlflow.pytorch.load_model(f"runs:/{latest_run_id}/models")
        print(loaded_model)
        # Predict:
        predictions = []
        actuals = []
        for Xs, ys in test_loader:
            output = loaded_model(Xs)
            predictions.extend(output.flatten().tolist())
            actuals.extend(ys.flatten().tolist())
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        predictions_rescaled = out_scaler.inverse_transform(predictions.reshape(-1,1)).flatten()
        actuals_rescaled = out_scaler.inverse_transform(actuals.reshape(-1,1)).flatten()

        preds = pd.Series(predictions_rescaled, index=y_test[sequence_len-1:].index)
        obs = pd.Series(actuals_rescaled, index=y_test[sequence_len-1:].index)

        r2, mse, rmse, mae, mape = evaluate(preds, obs, verbose=True)
        
        # Plot
        plot(preds, obs)

        return preds, obs
    else:
        print("No runs found in the specified experiment.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate LSTM model")
    parser.add_argument("--config-file", "-c", type=str, default='configs/lstm.yaml')
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    eval(
        config["data"]["features"],
        config["data"]["sequence_len"],
        config['mlflow']['experiment_name'],
    )
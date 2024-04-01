import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from mlflow.models.signature import infer_signature
from src.models.LSTM.data import load_data
from src.models.LSTM.model import StockPriceLSTM
from src.misc import split_data, evaluate, create_sequences
from sklearn.preprocessing import StandardScaler

# torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


def train(
    features,
    sequence_len,
    hidden_dim,
    num_layers,
    num_epochs,
    batch_size,
    lr,
    output_path,
    tracking_uri,
    experiment_name,
    tags,
):
    # Load data
    X, y = load_data(features=features, sequence_len=sequence_len)

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)

    # Normalisation
    in_scaler = StandardScaler()
    out_scaler = StandardScaler()
    X_train_norm = in_scaler.fit_transform(X_train.values)
    X_val_norm = in_scaler.transform(X_val.values)
    X_test_norm = in_scaler.transform(X_test.values)
    y_train_norm = out_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_norm = out_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_norm = out_scaler.transform(y_test.values.reshape(-1, 1))

    # Sequencing
    X_train_seq, y_train_seq = create_sequences(
        X_train_norm, y_train_norm, sequence_len
    )
    X_val_seq, y_val_seq = create_sequences(X_val_norm, y_val_norm, sequence_len)
    X_test_seq, y_test_seq = create_sequences(X_test_norm, y_test_norm, sequence_len)

    # TensorDatasets and DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_seq.astype(np.float32)),
        torch.tensor(y_train_seq.astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_seq.astype(np.float32)),
        torch.tensor(y_val_seq.astype(np.float32)),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_seq.astype(np.float32)),
        torch.tensor(y_test_seq.astype(np.float32)),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # Shuffle or no shuffle?
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model:
    input_dim = 5  # Number of features
    output_dim = 1  # Number of output classes (predicting 'Close Forecast')
    model = StockPriceLSTM(input_dim, hidden_dim, num_layers, output_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Log parameters:
    params = {
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "lr": lr,
        "optimizer": "Adam",
        "loss": criterion.__class__.__name__,
        "epochs": num_epochs,
        "batch size": batch_size,
        "sequence length": sequence_len,
    }

    # Set tracking server uri for logging
    mlflow.set_tracking_uri(tracking_uri)
    # Create an MLflow Experiment
    mlflow.set_experiment(experiment_name)
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Train:
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for Xs, ys in train_loader:
                optimizer.zero_grad()
                outputs = model(Xs)
                loss = criterion(outputs, ys)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= Xs.size(0)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xs, ys in val_loader:
                    outputs = model(Xs)
                    loss = criterion(outputs, ys)
                    val_loss += loss.item()
            val_loss /= Xs.size(0)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss}, Validation loss: {val_loss}"
            )

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch + 1
            )

        # Evaluate
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Inference without calculating gradients
            predictions = []
            actuals = []
            for Xs, ys in test_loader:
                output = model(Xs)
                predictions.extend(output.flatten().tolist())
                actuals.extend(ys.flatten().tolist())
            predictions = np.array(predictions)
            actuals = np.array(actuals)
        predictions_rescaled = out_scaler.inverse_transform(predictions.reshape(-1, 1))
        actuals_rescaled = out_scaler.inverse_transform(actuals.reshape(-1, 1))
        r2, mse, rmse, mae, mape = evaluate(predictions_rescaled, actuals_rescaled)

        # Log the metrics
        mlflow.log_metrics(
            {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
        )  # type: ignore

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tags(tags)

        # Save model:
        inputs, _ = next(iter(train_loader))
        with torch.no_grad():
            outputs = model(inputs)
        signature = infer_signature(
            inputs.detach().cpu().numpy(), outputs.detach().cpu().numpy()
        )
        mlflow.pytorch.log_model(model, output_path, signature=signature)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument("--config-file", "-c", type=str, default="configs/lstm.yaml")
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    train(
        config["data"]["features"],
        config["data"]["sequence_len"],
        config["model"]["hidden_dim"],
        config["model"]["num_layers"],
        config["training"]["num_epochs"],
        config["training"]["batch_size"],
        config["training"]["lr"],
        config["training"]["output_path"],
        config["mlflow"]["tracking_uri"],
        config["mlflow"]["experiment_name"],
        config["mlflow"]["tags"],
    )

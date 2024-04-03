import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import StandardScaler
from src.misc import split_data, evaluate, create_sequences


from src.models.Transformer.data import load_data
from src.models.Transformer.model import TimeSeriesTransformer

np.random.seed(1)
torch.manual_seed(1)

def train(
    features,
    sequence_len,
    num_layers,
    dropout,
    num_epochs,
    lr,
    experiment_name,
):
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

    # Instantiate the model
    model = TimeSeriesTransformer(feature_size=len(features), num_layers=num_layers, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Log parameters:
    params = {
        "features": features,
        "sequence_len": sequence_len,
        "num_layers": num_layers,
        "dropout": dropout,
        "lr": lr,
        "optimizer": optimizer.__class__.__name__,
        "loss": criterion.__class__.__name__,
        "num_epochs": num_epochs,
    }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)

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

            # # Validation
            # model.eval()
            # val_loss = 0.0
            # with torch.no_grad():
            #     for Xs, ys in val_loader:
            #         outputs = model(Xs)
            #         loss = criterion(outputs, ys)
            #         val_loss += loss.item()
            # val_loss /= Xs.size(0)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Training loss: {train_loss}, Validation loss: n/a"
            )

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": 0}, step=epoch + 1
            )

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
        r2, mse, rmse, mae, mape = evaluate(predictions, actuals)

        # Log the metrics
        mlflow.log_metrics(
            {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
        )  # type: ignore

        # Save model:
        inputs, _ = next(iter(train_loader))
        with torch.no_grad():
            outputs = model(inputs)
        signature = infer_signature(
            inputs.detach().cpu().numpy(), outputs.detach().cpu().numpy()
        )
        mlflow.pytorch.log_model(model, "model", signature=signature)


if __name__ == "__main__":
    with open("configs/transformer.yaml", "r") as file:
        config = yaml.safe_load(file)

    train(
        config["data"]["features"],
        config["data"]["sequence_len"],
        config["model"]["num_layers"],
        config["model"]["dropout"],
        config["training"]["num_epochs"],
        config["training"]["lr"],
        config["mlflow"]["experiment_name"],
    )

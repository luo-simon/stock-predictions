import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from mlflow.models.signature import infer_signature
from src.models.LSTM.data import load_data, denorm
from src.models.LSTM.model import StockPriceLSTM
from src.misc import split_data, evaluate

# torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def train(features, num_epochs, batch_size, output_path, tracking_uri, experiment_name, tags):
    # Load data
    X, y = load_data(features=features, sequence_len=60)
    
    # Split
    X_train, X_val, X_test = split_data(X, verbose=True)
    y_train, y_val, y_test = split_data(y, verbose=True)

    # TensorDatasets + DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffle or no shuffle?
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model:
    input_dim = 5  # Number of features
    hidden_dim = 64  # Number of features in hidden state
    num_layers = 2  # Number of stacked LSTM layers
    output_dim = 1  # Number of output classes (predicting 'Close Forecast')

    model = StockPriceLSTM(input_dim, hidden_dim, num_layers, output_dim)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train:
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Make sure labels are the correct shape
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Log parameters:
    params = {
        "hidden_dim": 64,
        "num_layers": 2,
        "lr": 0.001, 
        "optimizer": "Adam",
        "loss": "MSE",
        "epochs": num_epochs,
        "batch size": batch_size
    }

    # Evaluate
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Inference without calculating gradients
        predictions = []
        actuals = []
        for inputs, labels in test_loader:
            output = model(inputs)
            predictions.extend(output.view(-1).tolist())
            actuals.extend(labels.tolist())
        predictions = np.array(predictions)
        actuals = np.array(actuals)

    predictions_rescaled = denorm(predictions)
    actuals_rescaled = denorm(actuals)

    r2, mse, rmse, mae, mape = evaluate(predictions_rescaled, actuals_rescaled)

    # Set tracking server uri for logging
    mlflow.set_tracking_uri(tracking_uri)
    # Create an MLflow Experiment
    mlflow.set_experiment(experiment_name)
    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)
        # Log the metrics
        mlflow.log_metrics({
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        })
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tags(tags)

        # Save model:
        signature = infer_signature(X_train, y_train)
        mlflow.pytorch.log_model(model, output_path, signature=signature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument("--config-file", "-c", type=str, default='configs/lstm.yaml')
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    train(
        config["data"]["features"],
        config["training"]["num_epochs"],
        config['training']['batch_size'],
        config['training']['output_path'],
        config['mlflow']['tracking_uri'],
        config['mlflow']['experiment_name'],
        config['mlflow']['tags']
    )
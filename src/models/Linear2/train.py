from src.models.Linear2.data import load_data
from src.models.Linear2.evaluate import eval
from src.misc import split_data, evaluate, plot
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature


def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

def train():
    # Enable MLflow autologging
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("linear")
    mlflow.sklearn.autolog()

    # Load dataset
    X, y = load_data()

    # Split
    X_train, X_val, X_test = split_data(X, verbose=True)
    y_train, y_val, y_test = split_data(y, verbose=False)

    # Train
    model = LinearRegression(fit_intercept=True)

    with mlflow.start_run() as run:
        model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # Evaluate
    preds = model.predict(X_test)
    r2, mse, rmse, mae, mape = evaluate(preds, y_test)

    # Log the metrics
    mlflow.log_metrics({
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    })

    # Save model:
    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(model, "models", signature=signature)


if __name__ == '__main__':
    train()
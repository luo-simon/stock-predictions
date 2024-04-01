from src.models.Linear.data import load_data
from src.misc import split_data, evaluate
import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
from mlflow.models.signature import infer_signature


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

    with mlflow.start_run() as _:
        model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    # Evaluate
    preds = model.predict(X_test)
    r2, mse, rmse, mae, mape = evaluate(preds, y_test)

    # Log the metrics
    mlflow.log_metrics({"r2": r2, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape})

    # Save model:
    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(model, "models", signature=signature)


if __name__ == "__main__":
    train()

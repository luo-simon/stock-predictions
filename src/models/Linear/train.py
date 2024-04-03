import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from src.misc import split_data, evaluate
import matplotlib.pyplot as plt
import numpy as np

from src.models.Linear.data import load_data
from sklearn.linear_model import LinearRegression


def train(features=[]):
    # Load dataset
    X, y = load_data()

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])

    if len(features) > 0:
        X_train = X_train[features]
        X_test = X_test[features]

    # Model
    model = LinearRegression(fit_intercept=True)

    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("linear_new")
    with mlflow.start_run():
        # Log params
        mlflow.log_params({"features": features})

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        r2, mse, rmse, mae, mape = evaluate(preds, y_test)

        # Log the metrics
        mlflow.log_metrics(
            {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
        )

        # # Plot
        features_coeffs = np.vstack((X_train.columns, model.coef_)).T
        features_coeffs = sorted(
            features_coeffs, key=lambda x: abs(float(x[1])), reverse=False
        )
        feature_names, coeffs = zip(*features_coeffs)

        plt.figure(figsize=(12, 10))
        plt.barh(feature_names, coeffs)
        plt.ylabel("Features")
        plt.xlabel("Coefficient Value")
        plt.title("Feature Coefficients from Linear Regression")
        plot_path = "/Users/simon/Documents/II/Dissertation/figures/linear.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        # Save model:
        signature = infer_signature(X_train, preds)
        mlflow.sklearn.log_model(model, "model", signature=signature)


if __name__ == "__main__":
    features = [
        "Close",
        "SMA_5",
        "williams_r",
        "TRANGE",
        "AD",
        "EMA_50",
        "log_^FTSE",
        "log_low",
        "upper_band",
    ]
    train(features=features)

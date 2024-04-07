import pandas as pd
import mlflow
from src.misc import split_data, evaluate
import matplotlib.pyplot as plt
import numpy as np

from src.models.Linear_L1.data import load_data
from src.models.Linear_L1.model import Lasso


def train(alpha=0.1):
    # Enable MLflow autologging
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("linear_l1")
    mlflow.sklearn.autolog()

    # Load dataset
    X, y = load_data()

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)
    X_train = pd.concat([X_train, X_val])
    y_train = pd.concat([y_train, y_val])

    # Model
    model = Lasso(alpha=alpha)

    with mlflow.start_run() as _:
        # Log params
        mlflow.log_params({"alpha": alpha})

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        preds = model.predict(X_test)
        r2, mse, rmse, mae, mape = evaluate(preds, y_test)

        # Log the test metrics
        mlflow.log_metrics(
            {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
        )

        # Plot
        features_coeffs = np.vstack((X_train.columns, model.coef_)).T
        features_coeffs = sorted(
            features_coeffs, key=lambda x: abs(float(x[1])), reverse=False
        )
        features, coeffs = zip(*features_coeffs)

        plt.figure(figsize=(12, 10))
        plt.barh(features, coeffs)
        plt.ylabel("Features")
        plt.xlabel("Coefficient Value")
        plt.title("Feature Coefficients from Lasso Regression")
        plot_path = "/Users/simon/Documents/II/Dissertation/figures/linear_l1.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)


if __name__ == "__main__":
    train(alpha=0.1)

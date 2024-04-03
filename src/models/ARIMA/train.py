from src.models.ARIMA.data import load_data
from src.misc import split_data, evaluate, plot
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
import mlflow

def train(p, d, q, experiment_name):
    # Load dataset
    X, y = load_data()

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)

    # Model
    model = ARIMA(y_train, order=(p,d,q))
    mlflow.statsmodels.autolog()
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Log params
        mlflow.log_params({"p":p, "d":d, "q":q})

        # Train
        fit_res = model.fit()

        # Evaluate on validation set
        model = ARIMA(y, order=(p,d,q))
        res = model.filter(fit_res.params)
        predict = res.get_prediction()
        # predict_ci = predict.conf_int() # todo: conf. intervals
        preds = predict.predicted_mean
        r2, mse, rmse, mae, mape = evaluate(preds.loc[y_val.index], y_val)
        # Log the metrics
        mlflow.log_metrics(
            {"val_rmse": rmse}
        )

        # Evaluate on test set
        r2, mse, rmse, mae, mape = evaluate(preds.loc[y_test.index], y_test)
        # Log the metrics
        mlflow.log_metrics(
            {"test_rmse": rmse}
        )
    


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train LSTM model")
    # parser.add_argument("--config-file", "-c", type=str, default='configs/lstm.yaml')
    # args = parser.parse_args()

    # # Load the configuration file:
    # with open(args.config_file, 'r') as file:
    #     config = yaml.safe_load(file)

    train(1,1,1, "arima")

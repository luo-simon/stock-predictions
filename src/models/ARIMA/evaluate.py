from src.models.ARIMA.data import load_data
from src.misc import split_data, evaluate, plot, load_model_from_run_id, print_metrics
import mlflow
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def eval(p, d, q, run_id, stocks=["aapl"]):
    metrics = np.array([0.,0.,0.,0.,0.])
    for stock in stocks:
        # Load test data
        X, y = load_data(stock)

        # Split
        _, _, X_test = split_data(X, verbose=False)
        _, _, y_test = split_data(y, verbose=False)

        # Load model
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        fit_res = mlflow.statsmodels.load_model(f"runs:/{run_id}/model")

        # Evaluate
        model = ARIMA(y, order=(p, d, q))
        res = model.filter(fit_res.params)
        predict = res.get_prediction()
        # predict_ci = predict.conf_int() # todo: conf. intervals
        preds = predict.predicted_mean.loc[y_test.index]

        metrics += np.array(evaluate(preds, y_test, verbose=True))

        # Plot
        plot(preds, y_test)

    metrics /= len(stocks)
    print_metrics(metrics)
    return metrics


if __name__ == "__main__":
    p, d, q = 2, 1, 3
    run_id = "b535292cf0aa4b67aceb43ac213e66f9"
    eval(p, d, q, run_id)

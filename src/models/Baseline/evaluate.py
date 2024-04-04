from src.models.Baseline.data import load_data
from src.misc import split_data, evaluate, plot, print_metrics
import numpy as np

def eval(stocks=["aapl"]):
    metrics = np.array([0.,0.,0.,0.,0.])
    # Load test data
    for stock in stocks:
        X, y = load_data(stock)

        # Split
        _, _, X_test = split_data(X, verbose=False)
        _, _, y_test = split_data(y, verbose=False)

        # Predict
        preds = X_test

        # Evaluate
        metrics += np.array(evaluate(preds, y_test, verbose=True))


        # Plot
        plot(preds, y_test, stock)
    metrics /= len(stocks)
    print_metrics(metrics)
    
    return preds, y_test


if __name__ == "__main__":
    eval()

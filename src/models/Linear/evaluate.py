from src.models.Linear.data import load_data
from src.misc import (
    split_data,
    evaluate,
    plot,
    load_model_from_run_id,
    print_metrics
)
import pandas as pd
import numpy as np


def eval(run_id, features, stocks=["aapl"]):
    metrics = np.array([0.,0.,0.,0.,0.])
    for stock in stocks:
        # Load dataset
        X, y = load_data(stock)
        X = X[features]

        # Split
        X_train, X_val, X_test = split_data(X, verbose=False)
        y_train, y_val, y_test = split_data(y, verbose=False)

        # Load model
        model = load_model_from_run_id(run_id, flavor="sklearn")

        # Predict
        preds = pd.Series(model.predict(X_test), index=y_test.index)

        # Evaluate
        metrics += np.array(evaluate(preds, y_test, verbose=True))
        
        # Plot
        plot(preds, y_test)

    metrics /= len(stocks)
    print_metrics(metrics)
    return preds, y_test

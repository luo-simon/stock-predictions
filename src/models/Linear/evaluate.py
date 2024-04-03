from src.models.Linear.data import load_data
from src.misc import (
    split_data,
    evaluate,
    plot,
    load_model_from_experiment, load_model_from_run_id
)
import pandas as pd


def eval():
    # Load dataset
    X, y = load_data()

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)

    # Load model
    model = load_model_from_run_id("59ee53bdc13b4141b7b2acc273d0879f", flavor="sklearn")

    # Predict
    preds = pd.Series(model.predict(X_test), index=y_test.index)

    # Evaluate
    r2, mse, rmse, mae, mape = evaluate(preds, y_test, verbose=True)

    # Plot
    plot(preds, y_test)

    return preds, y_test


if __name__ == "__main__":
    eval()

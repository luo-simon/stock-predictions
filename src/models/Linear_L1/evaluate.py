import pandas as pd
from src.misc import split_data, evaluate, plot, load_sklearn_model_from_latest_run

from src.models.Linear_L1.data import load_data


def eval():
    # Load dataset
    X, y = load_data()

    # Split
    X_train, X_val, X_test = split_data(X, verbose=False)
    y_train, y_val, y_test = split_data(y, verbose=False)

    # Load model
    model = load_sklearn_model_from_latest_run("linear")

    # Predict
    preds = pd.Series(model.predict(X_test), index=y_test.index)

    # Evaluate
    r2, mse, rmse, mae, mape = evaluate(preds, y_test, verbose=True)

    # Plot
    plot(preds, y_test)

    return preds, y_test


if __name__ == "__main__":
    eval()

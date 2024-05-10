import numpy as np
import pandas as pd
from functools import partial

from src.misc import compute_accuracy, get_study, load_processed_dataset
from src.parser import get_parser

from sklearn.linear_model import LinearRegression, Lasso, Ridge

np.random.seed(42)


def data(stock, feature_set):
    df = load_processed_dataset(
        stock, start_date=f"2004-01-01", end_date="2024-01-01"
    )
    drop_cols = [c for c in df.columns if "forecast" in c.lower()]
    X = df.drop(drop_cols, axis=1)[feature_set]
    y = df["log_return_forecast"]
    X_train, X_val, X_test = (
        X[:"2022-01-01"],
        X["2022-01-01":"2023-01-01"],
        X["2023-01-01":],
    )
    y_train, y_val, y_test = (
        y[:"2022-01-01"],
        y["2022-01-01":"2023-01-01"],
        y["2023-01-01":],
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def predict(stock, feature_set):
    X_train, y_train, X_val, y_val, X_test, y_test = data(stock, feature_set)
    model = LinearRegression(fit_intercept=True)
    # model = Lasso(fit_intercept=True, alpha=0.0003)
    # model = Ridge(fit_intercept=True, alpha=0.0003)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    val_df = pd.DataFrame(data={"Predictions": val_preds, "Actuals": y_val})
    test_df = pd.DataFrame(data={"Predictions": test_preds, "Actuals": y_test})

    # print(pd.Series(index=X_train.columns, data=model.coef_).sort_values())
    return val_df, test_df


def objective(trial, stock, feature_set):
    X_train, y_train, X_val, y_val, X_test, y_test = data(stock, feature_set)
    val_df, test_df = predict(stock, feature_set)
    return compute_accuracy(val_df["Predictions"], y_val)


if __name__ == "__main__":
    parser, parser_fit, parser_tune = get_parser()
    args = parser.parse_args()

    fixed_objective = partial(
        objective, stock=args.stock, feature_set=args.features
    )
    study = get_study(experiment_name=args.experiment_name)
    study.optimize(fixed_objective, n_trials=1)

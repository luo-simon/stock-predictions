import pandas as pd
import numpy as np
from functools import partial

from src.misc import get_study, load_processed_dataset, compute_accuracy
from src.parser import get_parser

from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

def data(stock, feature_set):
    df = load_processed_dataset(stock, start_date=f"2004-01-01", end_date="2024-01-01")
    drop_cols = [c for c in df.columns if "forecast" in c.lower()]
    X = df.drop(drop_cols, axis=1)[feature_set]
    y = df["log_return_forecast"]
    X_train, X_val, X_test = X[:"2022-01-01"], X["2022-01-01":"2023-01-01"], X["2023-01-01":]
    y_train, y_val, y_test = y[:"2022-01-01"], y["2022-01-01":"2023-01-01"], y["2023-01-01":]
    return X_train, y_train, X_val, y_val, X_test, y_test

def predict(stock, feature_set, **params):
    X_train, y_train, X_val, y_val, X_test, y_test = data(stock, feature_set)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)
    val_df = pd.DataFrame(data={"Predictions": val_preds, "Actuals": y_val})
    test_df = pd.DataFrame(data={"Predictions": test_preds, "Actuals": y_test})
    return val_df, test_df

def objective(trial, stock, feature_set):
    X_train, y_train, X_val, y_val, X_test, y_test = data(stock, feature_set)

    n_estimators = trial.suggest_int('n_estimators', 100, 1000, step=100)
    max_features = trial.suggest_float('max_features', 0, 1)
    max_depth = trial.suggest_categorical('max_depth', [10, 20, 30, None])
    min_samples_split = trial.suggest_categorical('min_samples_split', [2, 5, 10, 20])
    min_samples_leaf =  trial.suggest_categorical('min_samples_leaf', [1, 2, 5, 10, 20])
    bootstrap =  trial.suggest_categorical('bootstrap', [True, False])

    val_df, test_df = predict(
        stock,
        feature_set,
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap
    )

    return compute_accuracy(val_df["Predictions"], y_val)

if __name__ == "__main__":
    parser, parser_fit, parser_tune = get_parser()
    parser_fit.add_argument('--n_estimators', type=int, default=100)
    parser_fit.add_argument('--max_features', type=float, default=1.)
    parser_fit.add_argument('--max_depth', type=int, default=None)
    parser_fit.add_argument('--min_samples_split', type=int, default=2)
    parser_fit.add_argument('--min_samples_leaf', type=int, default=1)
    parser_fit.add_argument('--bootstrap', type=bool, default=True)
    args = parser.parse_args()

    fixed_objective = partial(objective, stock=args.stock, feature_set=args.features)
    study = get_study(experiment_name=args.experiment_name)

    if args.subcommand == "fit":
        study.enqueue_trial(dict(args.fit))
        study.optimize(fixed_objective, n_trials=1)
    elif args.subcommand == "tune":
        study.optimize(fixed_objective, n_trials=args.tune.n_trials)
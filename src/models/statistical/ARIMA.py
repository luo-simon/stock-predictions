from jsonargparse import Namespace
import pandas as pd
import numpy as np
from functools import partial
import warnings

from src.misc import get_study, load_processed_dataset, compute_accuracy
from src.parser import get_parser

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

np.random.seed(42)
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")

def data(stock, feature_set):
    df = load_processed_dataset(stock, start_date=f"2004-01-01", end_date="2024-01-01")
    endog = df["log_return"].to_period('D')
    exog = df[feature_set].drop(columns=["log_return"]).to_period('D')
    y = df["log_return_forecast"]
    return exog, endog, y

def predict(stock, feature_set, p, q):
    exog, endog, y = data(stock, feature_set)
    if exog.shape[1] == 0:
        model = sm.tsa.statespace.SARIMAX(endog=endog[:"2022-01-01"], exog=None, order=(p, 0, q), enforce_invertibility=False)
        exog=None
    else:
        model = sm.tsa.statespace.SARIMAX(endog=endog[:"2022-01-01"], exog=exog[:"2022-01-01"], order=(p, 0, q), enforce_invertibility=False)
    
    fitted_model = model.fit(disp=False)
    # Full dataset
    model = sm.tsa.statespace.SARIMAX(endog=endog, exog=exog, order=(p,0,q))
    res = model.filter(fitted_model.params)
    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    preds = pd.Series(predict.predicted_mean.values, name="Predictions", index=y.index)
    df = pd.DataFrame(data={"Predictions": preds, "Actuals": y})

    return df["2022-01-01":"2023-01-01"], df["2023-01-01": "2024-01-01"]

def objective(trial, args: Namespace):
    exog, endog, y = data(args.stock, args.features)
    p = trial.suggest_int('p', 0, 20)
    q = trial.suggest_int('q', 0, 20)
    val_df, test_df = predict(args.stock, args.features, p, q)
    trial.set_user_attr("config", str(vars(args)))
    return compute_accuracy(val_df["Predictions"], y["2022-01-01":"2023-01-01"])

if __name__ == "__main__":
    parser, parser_fit, parser_tune = get_parser()
    parser_fit.add_argument('-p', type=int, help='p', default=5)
    parser_fit.add_argument('-q', type=int, help='q', default=5)
    args = parser.parse_args()

    fixed_objective = partial(objective, args=args)
    study = get_study(experiment_name=args.experiment_name)

    if args.subcommand == "fit":
        study.enqueue_trial(dict(args.fit))
        study.optimize(fixed_objective, n_trials=1)
    elif args.subcommand == "tune":
        study.optimize(fixed_objective, n_trials=args.tune.n_trials)

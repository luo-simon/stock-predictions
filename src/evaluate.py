import torch
import optuna
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson
from lightning.pytorch.trainer.trainer import Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.misc import (
    load_processed_dataset,
    load_trial_from_experiment,
    compute_accuracy,
    load_best_n_trials_from_experiment,
    filter_stdout,
)


from src.models.CNN.model import CNNModel
from src.models.CNN.data import CNNDataModule
from src.models.LSTM.model import LSTMModel
from src.models.LSTM.data import LSTMDataModule
from src.models.ConvLSTM.model import ConvLSTMModel
from src.models.ConvLSTM.data import ConvLSTMDataModule

import src.models.statistical.Linear as Linear
import src.models.statistical.ARIMA as ARIMA
import src.models.statistical.RandomForest as RF

np.random.seed(42)
pd.options.display.float_format = "{:,.8f}".format


def get_ML_metrics(predicted: pd.Series, actuals: pd.Series, verbose=False):
    """
    Returns tuple of (r2, mse, rmse, mae, corr)
    """
    assert isinstance(predicted, pd.Series), "Predicted is not Pandas Series"
    assert isinstance(actuals, pd.Series), "Actuals is not Pandas Series"

    r2 = r2_score(actuals, predicted)
    mse = mean_squared_error(actuals, predicted)
    rmse = np.sqrt(mean_squared_error(actuals, predicted))
    mae = mean_absolute_error(actuals, predicted)
    corr = predicted.corr(actuals) if len(predicted.unique()) > 1 else "n/a"
    if verbose:
        print(f"{'R^2:':>10} {r2}")
        print(f"{'MSE:':>10} {mse}")
        print(f"{'RMSE:':>10} {rmse}")
        print(f"{'MAE:':>10} {mae}")
        print(f"{'p:':>10} {corr}")

    return r2, mse, rmse, mae, corr


def backtest(preds_series, actuals_series, verbose=False):
    action = np.where(preds_series <= 0, "hold", "buy")
    returns = np.exp((actuals_series * np.where(action == "hold", 0, 1))) - 1
    cum_return = (1 + returns).cumprod().iloc[-1] - 1
    avg_daily_return = (cum_return + 1) ** (1 / len(preds_series)) - 1
    std = returns.std()

    if verbose:
        print(
            f"Avg. Daily Return: {avg_daily_return}. Cumulative return: {cum_return}. Standard deviation: {std}"
        )

    return avg_daily_return, std


def calculate_future_prices(row, horizon=5):
    future_prices = []
    cumsum = 0
    cur_date = row.name
    for t in range(horizon):
        pred_date = cur_date + pd.Timedelta(days=t)
        cumsum += row.at[pred_date, "Predictions"]
        pred = row["Close"] * np.exp(cumsum)
        future_prices.append(pred)
    return future_prices


def predict(trial: optuna.trial.FrozenTrial, stock, feature_set):
    filter_stdout()

    "Returns a dataframe with predicted and true values for the model of a given Optuna trial"
    # ckpt = trial.user_attrs["ckpt_path"]
    ckpt = trial.user_attrs["last_ckpt_path"]
    data_hparams = trial.user_attrs["data_hparams"]
    model_type = trial.user_attrs["model"]

    name_map = {
        "CNN": (CNNModel, CNNDataModule),
        "LSTM": (LSTMModel, LSTMDataModule),
        "ConvLSTM": (ConvLSTMModel, ConvLSTMDataModule),
    }

    model = name_map[model_type][0].load_from_checkpoint(ckpt)
    dm = name_map[model_type][1](**data_hparams)
    trainer = Trainer(enable_progress_bar=False, enable_model_summary=False, logger=False)

    data = load_processed_dataset(
        stock, start_date="2022-01-01", end_date="2024-01-01"
    )
    validation_set = data[:"2023-01-01"]
    test_set = data["2023-01-01":]

    val_preds = trainer.predict(model, dataloaders=dm.val_dataloader())
    val_preds = torch.cat(val_preds, dim=0).squeeze().cpu().detach().numpy()
    val_preds = pd.Series(
        val_preds, index=validation_set.index, name="Predictions"
    )

    test_preds = trainer.predict(model, dm)
    test_preds = torch.cat(test_preds, dim=0).squeeze().cpu().detach().numpy()
    test_preds = pd.Series(
        test_preds, index=test_set.index, name="Predictions"
    )

    val_df = validation_set.join(val_preds)
    test_df = test_set.join(test_preds)
    val_df = val_df.rename({"log_return_forecast": "Actuals"}, axis=1)
    test_df = test_df.rename({"log_return_forecast": "Actuals"}, axis=1)

    return (
        test_df[["Predictions", "Actuals"]],
        val_df[["Predictions", "Actuals"]],
    )


def permutation_importance(model_type, stock):
    filter_stdout()
    experiment_name = f"{model_type}_{stock}"
    trial = load_best_n_trials_from_experiment(experiment_name, n=1)[0]
    ckpt = trial.user_attrs["last_ckpt_path"]
    data_hparams = trial.user_attrs["data_hparams"]
    model_type = trial.user_attrs["model"]

    name_map = {
        "CNN": (CNNModel, CNNDataModule),
        "LSTM": (LSTMModel, LSTMDataModule),
        "ConvLSTM": (ConvLSTMModel, ConvLSTMDataModule),
    }

    model = name_map[model_type][0].load_from_checkpoint(ckpt)
    dm = name_map[model_type][1](**data_hparams)
    trainer = Trainer(enable_progress_bar=False, enable_model_summary=False, logger=False)

    y_test = load_processed_dataset(
        stock, start_date="2023-01-01", end_date="2024-01-01"
    )["log_return_forecast"]
    y_test = y_test.rename("Actuals")

    features = [
        "log_return",
        "log_return_open",
        "log_return_high",
        "log_return_low",
        "log_return_volume",
        "sma",
        "wma",
        "ema",
        "dema",
        "tema",
        "aroon",
        "rsi",
        "willr",
        "cci",
        "ad",
        "mom",
        "slowk",
        "slowd",
        "macd",
        "fed_funds_rate",
        "^N225",
        "^IXIC",
        "^FTSE",
        "^SPX",
        "^DJI",
    ]

    df = []
    for i, f in enumerate(features):
        test_dataloader = dm.get_permuted_test_set(i)
        preds = trainer.predict(model, dataloaders=[test_dataloader])
        preds = torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()
        preds = pd.Series(preds, index=y_test.index, name="Predictions")
        metrics = get_all_metrics(preds, y_test)
        df.append(metrics)
    df = pd.DataFrame(df, index=features)[
        ["RMSE", "Accuracy", "Avg. daily return", "Risk adj. return"]
    ]
    baseline = get_results_df(experiment_name)["Test set"].iloc[0][
        ["RMSE", "Accuracy", "Avg. daily return", "Risk adj. return"]
    ]
    return df - baseline


def get_prediction_dfs_from_experiment(experiment_name, trial_num=None):
    model_type = experiment_name.split("_")[0]
    stock = experiment_name.split("_")[1]

    if trial_num is not None:
        trial = load_trial_from_experiment(experiment_name, trial_num)
    else:
        trial = load_best_n_trials_from_experiment(experiment_name, n=1)[0]

    feature_set = trial.user_attrs.get("feature_set", None)
    if not feature_set:
        feature_set = [
            "log_return",
            "log_return_open",
            "log_return_high",
            "log_return_low",
            "log_return_volume",
            "sma",
            "wma",
            "ema",
            "dema",
            "tema",
            "aroon",
            "rsi",
            "willr",
            "cci",
            "ad",
            "mom",
            "slowk",
            "slowd",
            "macd",
            "fed_funds_rate",
            "^N225",
            "^IXIC",
            "^FTSE",
            "^SPX",
            "^DJI",
        ]
    if model_type == "Linear":
        val_df, test_df = Linear.predict(stock, feature_set)
    elif model_type == "ARIMA":
        val_df, test_df = ARIMA.predict(stock, feature_set, **trial.params)
    elif model_type in ["RF", "RandomForest"]:
        val_df, test_df = RF.predict(stock, feature_set, **trial.params)
    elif model_type in ["CNN", "LSTM", "ConvLSTM"]:
        test_df, val_df = predict(trial, stock, feature_set)
    else:
        print(
            "Model type not recognised. Check experiment name is correctly named/typed."
        )

    hparams = trial.params
    return val_df, test_df, hparams


def get_results_df(experiment_name, trial_num=None):
    val_df, test_df, hparams = get_prediction_dfs_from_experiment(
        experiment_name, trial_num
    )

    val_metrics = get_all_metrics(val_df["Predictions"], val_df["Actuals"])
    val = pd.DataFrame(val_metrics, index=[experiment_name])
    val.columns = pd.MultiIndex.from_product([["Validation set"], val.columns])

    test_metrics = get_all_metrics(test_df["Predictions"], test_df["Actuals"])
    test = pd.DataFrame(test_metrics, index=[experiment_name])
    test.columns = pd.MultiIndex.from_product([["Test set"], test.columns])

    df = pd.concat([val, test], axis=1)
    df["Hyperparameters"] = str(hparams)

    return df


def visualise(preds, actuals):
    plt.rcParams.update({"font.size": 16})

    # PREDICTIONS PLOT in LOG RETURN (w/ 95% conf. interval)
    fig, ax = plt.subplots()
    residuals = actuals - preds
    interval = (
        1.96 * residuals.std()
    )  # 95% of area under a normal curve lives within ~1.95 std devs.
    lo = preds - interval
    hi = preds + interval
    ax.plot(actuals, label="Actual Log Return")
    ax.plot(preds, label="Predicted Log Return", color="orange")
    ax.fill_between(
        preds.index, lo.values, hi.values, color="orange", alpha=0.2
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.set_title("Actual and Predicted Log Return")
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    # PREDICTION PLOT in PRICE
    fig, ax = plt.subplots()
    actuals_price = np.exp(actuals.cumsum())
    preds_price = actuals_price.shift(1, fill_value=1) * np.exp(preds)
    price_residuals = actuals_price - preds_price
    lo = preds_price - interval
    hi = preds_price + interval
    interval = (
        1.96 * price_residuals.std()
    )  # 95% of area under a normal curve lives within ~1.95 std devs.
    ax.plot(actuals_price, label="Actual Price")
    ax.plot(preds_price, label="Predicted Price", color="orange")
    ax.fill_between(
        preds.index, lo.values, hi.values, color="orange", alpha=0.2
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Actual and Predicted Price")
    ax.xaxis.set_tick_params(rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    # CORRELATION PLOT
    fig, ax = plt.subplots()
    ax.scatter(actuals, preds, label="Predictions", s=0.5)
    ax.scatter(actuals, np.zeros_like(actuals), label="Naive", s=0.5)
    ax.plot(
        [actuals.min(), actuals.max()],
        [actuals.min(), actuals.max()],
        "r--",
        label="Perfect Prediction",
        linewidth=0.5,
    )
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Correlation Plot")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    # RESIDUALS PLOT
    fig, ax = plt.subplots()
    ax.plot(residuals, label="Residuals")
    ax.set_title("Plot of residuals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_tick_params(rotation=45)
    fig.tight_layout()

    # RESIDUALS DISTRIBUTION
    fig, ax = plt.subplots()
    sns.histplot(residuals, ax=ax, stat="density", kde=True)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax.plot(x, p, "k", linewidth=2, label="Normal", linestyle="dashed")
    ax.set_title("Distribution of residuals")
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    # ACF PLOT OF RESIDUALS
    fig, ax = plt.subplots()
    plot_acf(residuals, ax=ax, alpha=0.05, title="ACF Plot of Residuals")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    # QQ PLOT
    sorted_data = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(
        (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
    )
    res = stats.linregress(theoretical_quantiles, sorted_data)
    fig, ax = plt.subplots()
    ax.plot(
        theoretical_quantiles,
        sorted_data,
        "o",
        label="Residuals",
        markersize=2,
    )
    ax.plot(
        theoretical_quantiles,
        res.slope * theoretical_quantiles + res.intercept,
        "r-",
        label="Fit Line",
    )
    z = stats.norm.ppf(0.975)
    se = res.slope * np.sqrt(
        1 / len(sorted_data)
        + (theoretical_quantiles - np.mean(theoretical_quantiles)) ** 2
        / np.sum((theoretical_quantiles - np.mean(theoretical_quantiles)) ** 2)
    )
    ci = z * se
    ax.fill_between(
        theoretical_quantiles,
        res.slope * theoretical_quantiles + res.intercept - ci,
        res.slope * theoretical_quantiles + res.intercept + ci,
        color="lightgrey",
        alpha=0.5,
        label=f"95% CI",
    )
    ax.set_xlabel("Normal theoretical quantiles")
    ax.set_ylabel("Residuals quantiles")
    ax.set_title("Q-Q plot of residuals")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    # TEST ZERO MEAN
    print(f"Residuals mean: {residuals.mean()}")
    t_stat, p_value = stats.ttest_1samp(residuals, popmean=0)
    print(f"Zero-mean: one-sample t-test: t={t_stat}, p-value={p_value}")

    # TEST NORMALITY (Shapiro-Wilk test)
    stat, p = stats.shapiro(residuals)
    print(f"Normality: Shapiro-Wilk Test: Statistics={stat}, p-value={p}")

    # TEST AUTOCORRELATION (Durbin-Watson test)
    dw_stat = durbin_watson(residuals)
    print(f"Autocorrelation: Durbin-Watson statistic: {dw_stat}")

    # SHOW PLOTS
    plt.show()


def get_all_metrics(preds, actuals):
    # Traditional ML metrics
    r2, mse, rmse, mae, corr = get_ML_metrics(predicted=preds, actuals=actuals)
    acc = compute_accuracy(preds, actuals)

    # Financial metrics
    avg_daily_return, std = backtest(preds, actuals, verbose=False)
    risk_adj_return = None
    if std != 0:
        risk_adj_return = avg_daily_return / std

    return {
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "p": corr,
        "Accuracy": acc,
        "Avg. daily return": avg_daily_return,
        "Std. daily return": std,
        "Risk adj. return": risk_adj_return,
    }


def get_mean_std(stock, column, start="2004-01-01", end="2022-01-01"):
    df = load_processed_dataset(stock, start, end)[column]
    return df.mean(), df.std()


def random_walk(stock, n=1000):
    df = load_processed_dataset(stock, "2023-01-01", "2024-01-01")
    actuals = df["log_return_forecast"]
    dfs = []

    for i in range(n):
        mean, std = get_mean_std(stock, "log_return_forecast")
        random_preds = np.random.normal(loc=mean, scale=std, size=len(actuals))
        random_preds = pd.Series(random_preds, index=actuals.index)
        metrics = get_all_metrics(random_preds, actuals)
        df = pd.DataFrame(metrics, index=[i])
        df["Stock"] = stock
        dfs.append(df)
    return pd.concat(dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the best model of a given experiment"
    )
    parser.add_argument(
        "-n",
        "--name",
        help="name of the experiment to evaluate",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--trial_num",
        help="trial number of the experiment to evaluate (lowest validation loss trial selected if not specified)",
        type=int,
    )
    parser.add_argument(
        "-b",
        "--best",
        help="if True, evaluates the best model from trial (lowest validation loss) otherwise, if False, evaluates the last model ",
        default=True,
    )
    args = parser.parse_args()

    df = get_results_df(args.name, args.trial_num)
    print(df)
    df.to_clipboard()
    val_df, test_df, hparams = get_prediction_dfs_from_experiment(
        args.name, args.trial_num
    )
    visualise(test_df["Predictions"], test_df["Actuals"])

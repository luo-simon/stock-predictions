from audioop import avg
from sklearn.base import validate_parameter_constraints
import torch
import optuna
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from src.misc import load_processed_dataset, load_trial_from_experiment, compute_accuracy, load_best_n_trials_from_experiment
from statsmodels.graphics.tsaplots import plot_acf
from lightning.pytorch.trainer.trainer import Trainer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
feature_set =  ['log_return', 'log_return_open', 'log_return_high', 'log_return_low','log_return_volume', 'sma', 'wma', 'ema', 'dema','tema', 'aroon', 'rsi', 'willr', 'cci', 'ad', 'mom', 'slowk', 'slowd', 'macd', 'fed_funds_rate', '^N225', '^IXIC', '^FTSE', '^SPX', '^DJI']
    

def get_ML_metrics(predicted, actuals, verbose=False):
    """
    Returns tuple of (r2, mse, rmse, mae, corr)
    """
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
    cum_return = (1+returns).cumprod().iloc[-1] - 1
    avg_daily_return = (cum_return+1)**(1/len(preds_series)) - 1
    std = returns.std()
    
    if verbose:
        print(
            f"Avg. Daily Return: {avg_daily_return}. Cumulative return: {cum_return}. Standard deviation: {std}"
        )

    return avg_daily_return, std


def calculate_future_prices(row, horizon=5):
    cur_price = row["Close"]
    future_prices = []
    cumsum = 0
    cur_date = row.name
    for t in range(horizon):
        pred_date = cur_date + pd.Timedelta(days=t)
        cumsum += row.at[pred_date, "Predictions"]
        pred = row["Close"] * np.exp(cumsum)
        future_prices.append(pred)
    return future_prices


def predict(trial: optuna.trial.FrozenTrial, best):
    "Returns a dataframe with predicted and true values for the model of a given Optuna trial"
    ckpt = trial.user_attrs["ckpt_path"]
    if not best:
        ckpt = trial.user_attrs["last_ckpt_path"]
    data_hparams = trial.user_attrs["data_hparams"]
    model_type = trial.user_attrs["model"]
    stock = data_hparams["stock"]

    name_map = {
        "CNN": (CNNModel, CNNDataModule),
        "LSTM": (LSTMModel, LSTMDataModule),
        "ConvLSTM": (ConvLSTMModel, ConvLSTMDataModule),
    }

    model = name_map[model_type][0].load_from_checkpoint(ckpt)
    dm = name_map[model_type][1](**data_hparams)
    trainer = Trainer()
    if model_type == "LSTM":
        y = dm.y
        val_y = (
            y["2022-01-01":"2023-01-01"]
            .apply(lambda row: row.tolist(), axis=1)
            .rename("One-week Actuals")
        )
        test_y = (
            y["2023-01-01":]
            .apply(lambda row: row.tolist(), axis=1)
            .rename("One-week Actuals")
        )
        val_preds = trainer.predict(model, dataloaders=dm.val_dataloader())
        val_preds = torch.cat(val_preds, dim=0).squeeze().cpu().detach().numpy()
        val_preds = pd.DataFrame(val_preds, index=val_y.index)
        val_preds = (
            val_preds.apply(lambda row: row.tolist(), axis=1)
            .rename("One-week Predictions")
            .to_frame()
        )
        test_preds = trainer.predict(model, dm)
        test_preds = torch.cat(test_preds, dim=0).squeeze().cpu().detach().numpy()
        test_preds = pd.DataFrame(test_preds, index=test_y.index)
        test_preds = (
            test_preds.apply(lambda row: row.tolist(), axis=1)
            .rename("One-week Predictions")
            .to_frame()
        )

        val_df = val_preds.join(val_y)
        test_df = test_preds.join(test_y)
        val_df["Predictions"] = val_df["One-week Predictions"].apply(
            lambda x: x[0] if x else None
        )
        val_df["Actuals"] = val_df["One-week Actuals"].apply(
            lambda x: x[0] if x else None
        )
        test_df["Predictions"] = test_df["One-week Predictions"].apply(
            lambda x: x[0] if x else None
        )
        test_df["Actuals"] = test_df["One-week Actuals"].apply(
            lambda x: x[0] if x else None
        )
        return test_df, val_df
    elif model_type == "CNN":
        data = load_processed_dataset(
            stock, start_date="2022-01-01", end_date="2024-01-01"
        )
        validation_set = data[:"2023-01-01"]
        test_set = data["2023-01-01":]

        val_preds = trainer.predict(model, dataloaders=dm.val_dataloader())
        val_preds = (
            torch.cat(val_preds, dim=0).squeeze().cpu().detach().numpy()
        )  # flatten if more than one batch, then squeeze
        val_preds = pd.Series(val_preds, index=validation_set.index, name="Predictions")

        preds = trainer.predict(model, dm)
        preds = (
            torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()
        )  # flatten if more than one batch, then squeeze
        preds = pd.Series(preds, index=test_set.index, name="Predictions")
        val_df = validation_set.join(val_preds)
        val_df = val_df.rename({"log_return_forecast": "Actuals"}, axis=1)
        test_df = test_set.join(preds)
        test_df = test_df.rename({"log_return_forecast": "Actuals"}, axis=1)
    elif model_type == "ConvLSTM":
        data = load_processed_dataset(
            stock, start_date="2022-01-01", end_date="2024-01-01"
        )
        validation_set = data[:"2023-01-01"]
        test_set = data["2023-01-01":]

        val_preds = trainer.predict(model, dataloaders=dm.val_dataloader())
        val_preds = (
            torch.cat(val_preds, dim=0).squeeze().cpu().detach().numpy()
        )  # flatten if more than one batch, then squeeze
        val_preds = pd.Series(val_preds, index=validation_set.index, name="Predictions")

        preds = trainer.predict(model, dm)
        preds = (
            torch.cat(preds, dim=0).squeeze().cpu().detach().numpy()
        )  # flatten if more than one batch, then squeeze
        preds = pd.Series(preds, index=test_set.index, name="Predictions")
        val_df = validation_set.join(val_preds)
        val_df = val_df.rename({"log_return_forecast": "Actuals"}, axis=1)
        test_df = test_set.join(preds)
        test_df = test_df.rename({"log_return_forecast": "Actuals"}, axis=1)

    return test_df[["Predictions", "Actuals"]], val_df[["Predictions", "Actuals"]]


def mean_dfs(dfs):
    dfs = pd.concat(dfs)
    dfs = dfs.groupby(dfs.index).mean()
    return dfs

def main(experiment_name, trial_num=None, best=True):
    model_type = experiment_name.split("_")[0]
    stock =  experiment_name.split("_")[1]

    trial = None
    if model_type != "Linear":
        if trial_num != None:
            trial = load_trial_from_experiment(experiment_name, trial_num)
        else:
            trial = load_best_n_trials_from_experiment(experiment_name, n=1)[0]

    if model_type == "Linear":
        val_df, test_df = Linear.predict(stock)
    elif model_type == "ARIMA":
        val_df, test_df = ARIMA.predict(stock, trial.params["p"], trial.params["q"])
    elif model_type == "RF":
        val_df, test_df = RF.predict(stock, **trial.params)
    elif model_type == "CNN":
        test_df, val_df = predict(trial, best)
    elif model_type == "LSTM":
        test_df, val_df = predict(trial, best)
    elif model_type == "ConvLSTM":
        test_df, val_df = predict(trial, best)
    else:
        print("Model type not recognised. Check experiment name is correctly named/typed.")
        
    val_metrics =  get_all_metrics(val_df["Predictions"], val_df["Actuals"])
    val = pd.DataFrame(val_metrics, index=[experiment_name])
    val.columns = pd.MultiIndex.from_product([["Validation set"], val.columns])

    test_metrics =  get_all_metrics(test_df["Predictions"], test_df["Actuals"])
    test = pd.DataFrame(test_metrics, index=[experiment_name])
    test.columns = pd.MultiIndex.from_product([["Test set"], test.columns])

    df = pd.concat([val, test], axis=1)
    if trial:
        df["Hyperparameters"] = str(trial.params)

    return df

def get_all_metrics(preds, actuals):
    # Traditional ML metrics
    r2, mse, rmse, mae, corr = get_ML_metrics(predicted=preds, actuals=actuals)
    acc = compute_accuracy(preds, actuals)
    # Financial metrics
    avg_daily_return, std = backtest(preds, actuals, verbose=False)
    risk_adj_return = None
    if std != 0:
        risk_adj_return = avg_daily_return/std
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

def evaluate(test_df, val_df):
    # Metrics DF
    index_tuples = [
        ("Validation set", "Model"),
        ("Validation set", "Naïve"),
        ("Test set", "Model"),
        ("Test set", "Naïve"),
    ]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["", ""])
    column_tuples = [
        ("One-day ahead", "R2"),
        ("One-day ahead", "MSE"),
        ("One-day ahead", "RMSE"),
        ("One-day ahead", "MAE"),
        ("One-day ahead", "p"),
        ("One-week ahead", "R2"),
        ("One-week ahead", "MSE"),
        ("One-week ahead", "RMSE"),
        ("One-week ahead", "MAE"),
        ("One-week ahead", "p"),
    ]
    multi_columns = pd.MultiIndex.from_tuples(column_tuples)
    metrics_df = pd.DataFrame(index=multi_index, columns=multi_columns)
    zero_series_val = pd.Series(index=val_df.index, data=0)
    zero_series_test = pd.Series(index=test_df.index, data=0)
    # One-day ahead
    one_day_metrics = []
    one_day_metrics.append(
        get_ML_metrics(
            actuals=val_df["Actuals"], predicted=val_df["Predictions"], verbose=False
        )
    )
    one_day_metrics.append(
        get_ML_metrics(
            actuals=val_df["Actuals"], predicted=zero_series_val, verbose=False
        )
    )
    one_day_metrics.append(
        get_ML_metrics(
            actuals=test_df["Actuals"], predicted=test_df["Predictions"], verbose=False
        )
    )
    one_day_metrics.append(
        get_ML_metrics(
            actuals=test_df["Actuals"], predicted=zero_series_test, verbose=False
        )
    )
    one_day_metrics = pd.DataFrame(
        one_day_metrics, columns=["R2", "MSE", "RMSE", "MAE", "p"]
    )
    for key, values in one_day_metrics.items():
        metrics_df[("One-day ahead", key)] = values.values

    # Trading df
    index_tuples = [
        ("Validation set", "Model"),
        ("Validation set", "Buy-and-hold"),
        ("Test set", "Model"),
        ("Test set", "Buy-and-hold"),
    ]
    multi_index = pd.MultiIndex.from_tuples(index_tuples)
    trading_df = pd.DataFrame(
        index=multi_index, columns=["PnL", "Std of returns", "Accuracy"]
    )
    pnl_std = []
    pnl_std.append(backtest(val_df["Predictions"], val_df["Actuals"]))
    pnl_std.append(
        backtest(val_df["Predictions"], val_df["Actuals"], buy_and_hold=True)
    )
    pnl_std.append(backtest(test_df["Predictions"], test_df["Actuals"]))
    pnl_std.append(
        backtest(test_df["Predictions"], test_df["Actuals"], buy_and_hold=True)
    )
    pnl_std = pd.DataFrame(pnl_std, columns=["PnL", "Std of returns"])
    for key, values in pnl_std.items():
        trading_df[key] = values.values
    accuracies = []
    accuracies.append(compute_accuracy(val_df["Predictions"], val_df["Actuals"]))
    accuracies.append(compute_accuracy(np.ones_like(val_df["Predictions"]), val_df["Actuals"]))
    accuracies.append(compute_accuracy(test_df["Predictions"], test_df["Actuals"]))
    accuracies.append(compute_accuracy(np.ones_like(test_df["Predictions"]), test_df["Actuals"]))
    trading_df["Accuracy"] = accuracies

    random_df = pd.DataFrame(
        index=["Validation set", "Test set"],
        columns=[
            "Mean PnL",
            "PnL Std.",
            "Mean Returns Std.",
            "Mean Accuracy",
            "Accuracy Std",
        ],
    )
    random_df = []

    fig = plt.figure(figsize=(15, 8))
    grid = plt.GridSpec(3, 2, hspace=0.4, wspace=0.1)
    val_ax = plt.subplot(grid[0, 0])
    test_ax = plt.subplot(grid[0, 1])
    val_acc_ax = plt.subplot(grid[1, 0])
    test_acc_ax = plt.subplot(grid[1, 1])
    axs = [val_ax, test_ax, val_acc_ax, test_acc_ax]
    titles = ["Validation set", "Test set"]
    for i, df in enumerate([val_df, test_df]):
        random_pnls = []
        random_stds = []
        random_accs = []
        for _ in range(10000):
            random_choice = np.random.choice([-1, 1], size=len(df))
            pnl, std = backtest(random_choice, df["Actuals"])
            acc = (np.sign(df["Actuals"]) == random_choice).astype(int).mean() * 100
            random_pnls.append(pnl)
            random_stds.append(std)
            random_accs.append(acc)
        mean_pnl = np.array(random_pnls).mean()
        pnl_std = np.array(random_pnls).std()
        mean_std = np.array(random_stds).mean()
        mean_acc = np.array(random_accs).mean()
        acc_std = np.array(random_accs).std()
        random_df.append([mean_pnl, pnl_std, mean_std, mean_acc, acc_std])

        # Plot
        sns.histplot(
            random_pnls, kde=True, ax=axs[i], stat="density", alpha=0.2, bins=30
        )
        xmin, xmax = axs[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(random_pnls), np.std(random_pnls))
        axs[i].plot(
            x, p, "k", linewidth=1, label="Normal Distribution", linestyle="dashed"
        )
        axs[i].set_title(f"{titles[i]} Distribution of PnLs for Random Walk")
        axs[i].set_xlabel("PnL")
        axs[i].set_ylabel("Density")
        model_pnl = trading_df.loc[(titles[i], "Model"), "PnL"]
        axs[i].axvline(x=model_pnl, color="r", label="Model PnL", linewidth=2)
        buy_hold_pnl = trading_df.loc[(titles[i], "Buy-and-hold"), "PnL"]
        axs[i].axvline(x=buy_hold_pnl, color="g", label="Buy-and-hold PnL", linewidth=2)
        axs[i].legend()
        # kde_line = axs[i].get_lines()[0].get_data()
        # kde_x, kde_y = kde_line
        # ci_high = np.percentile(kde_x, 95)
        ci_high = stats.norm.ppf(0.95, np.mean(random_pnls), np.std(random_pnls))
        axs[i].fill_between(x, p, where=(x >= ci_high), color="gray", alpha=0.5)
        axs[i].axvline(ci_high, color="gray", linestyle="--")

        sns.histplot(
            random_accs, kde=True, ax=axs[i + 2], stat="density", alpha=0.2, bins=30
        )
        xmin, xmax = axs[i + 2].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, np.mean(random_accs), np.std(random_accs))
        axs[i + 2].plot(
            x, p, "k", linewidth=1, label="Normal Distribution", linestyle="dashed"
        )
        axs[i + 2].set_title(f"{titles[i]} Distribution of Accuracies for Random Walk")
        axs[i + 2].set_xlabel("Accuracy")
        axs[i + 2].set_ylabel("Density")
        model_acc = trading_df.loc[(titles[i], "Model"), "Accuracy"]
        axs[i + 2].axvline(x=model_acc, color="r", label="Model Accuracy", linewidth=2)
        buy_hold_acc = trading_df.loc[(titles[i], "Buy-and-hold"), "Accuracy"]
        axs[i + 2].axvline(
            x=buy_hold_acc, color="g", label="Buy-and-hold Accuracy", linewidth=2
        )
        axs[i + 2].legend()
        # kde_line = axs[i+2].get_lines()[0].get_data()
        # kde_x, kde_y = kde_line
        # ci_high = np.percentile(kde_x, 95)
        ci_high = stats.norm.ppf(0.95, np.mean(random_accs), np.std(random_accs))
        axs[i + 2].fill_between(x, p, where=(x >= ci_high), color="gray", alpha=0.5)
        axs[i + 2].axvline(ci_high, color="gray", linestyle="--")

    random_df = pd.DataFrame(
        random_df,
        index=["Validation set", "Test set"],
        columns=[
            "Mean PnL",
            "PnL Std.",
            "Mean Returns Std.",
            "Mean Accuracy",
            "Accuracy Std",
        ],
    )

    print(metrics_df, "\n")
    print(trading_df, "\n")
    print(random_df, "\n")

    # Conversion back into price
    initial_price = 1
    test_df["Actual Close Forecast"] = initial_price * np.exp(test_df["Actuals"].cumsum())
    test_df["Close"] = test_df["Actual Close Forecast"].shift(1, fill_value=initial_price)
    val_df["Actual Close Forecast"] = initial_price * np.exp(val_df["Actuals"].cumsum())
    val_df["Close"] = val_df["Actual Close Forecast"].shift(1, fill_value=initial_price)

    # test_df["Predicted Cumsum T+1"] = (
    #     test_df["Predictions"].rolling(window=2).sum().shift(-1)
    # )
    # test_df["Predicted Cumsum T+2"] = (
    #     test_df["Predictions"].rolling(window=3).sum().shift(-2)
    # )

    # test_df["Predicted Close T+1"] = test_df["Close"] * np.exp(
    #     test_df["Predicted Cumsum T+1"]
    # )
    # test_df["Predicted Close T+2"] = test_df["Close"] * np.exp(
    #     test_df["Predicted Cumsum T+2"]
    # )

    test_df["Predicted Close T+0"] = test_df["Close"] * np.exp(test_df["Predictions"])
    val_df["Predicted Close T+0"] = val_df["Close"] * np.exp(val_df["Predictions"])
    
    # Priec Plots
    prices_ax = plt.subplot(grid[2, 1])
    prices_ax.plot(test_df["Actual Close Forecast"], label="Actual Close Forecast", linewidth=0.5)
    prices_ax.plot(test_df["Predicted Close T+0"], label="Predicted", color="red", linewidth=0.5)
    prices_ax.plot(test_df["Close"], label="Naive", color="lightgreen", linewidth=0.5)
    # for index, row in test_df.iterrows():
    #     prices_ax.plot(
    #         pd.date_range(start=index, periods=3, freq="B"),
    #         [row[f"Predicted Close T+{i}"] for i in range(3)],
    #         alpha=0.5,
    #     )
    prices_ax.set_xlabel("Date")
    prices_ax.set_ylabel("Price")
    prices_ax.legend()
    
    prices_ax = plt.subplot(grid[2, 0])
    prices_ax.plot(val_df["Actual Close Forecast"], label="Actual Close Forecast", linewidth=0.5)
    prices_ax.plot(val_df["Predicted Close T+0"], label="Predicted", color="red", linewidth=0.5)
    prices_ax.plot(val_df["Close"], label="Naive", color="lightgreen", linewidth=0.5)
    prices_ax.set_xlabel("Date")
    prices_ax.set_ylabel("Price")
    prices_ax.legend()

    grid.tight_layout(fig)

    # Residuals
    fig = plt.figure(figsize=(15, 8))
    residuals = test_df["Actuals"] - test_df["Predictions"]
    val_residuals = val_df["Actuals"] - val_df["Predictions"]

    # Prediction Intervals
    interval = 1.96 * val_residuals.std()
    test_df["lo"] = test_df["Predictions"] - interval
    test_df["hi"] = test_df["Predictions"] + interval

    print(test_df.sample(5))
    grid = plt.GridSpec(2, 3)

    # Actuals and Predicted Time Series
    preds_ax = plt.subplot(grid[0, 0])
    preds_ax.plot(test_df.index, test_df["Actuals"], label="Actuals")
    preds_ax.plot(
        test_df.index, test_df["Predictions"], label="Predictions", color="orange"
    )
    preds_ax.fill_between(
        test_df.index, test_df["lo"], test_df["hi"], color="orange", alpha=0.2
    )
    preds_ax.set_xlabel("Date")
    preds_ax.set_ylabel("Values")
    preds_ax.set_title("Actual and Predicted Time Series ")
    preds_ax.legend()
    preds_ax.grid(True, alpha=0.2)

    corr_ax = plt.subplot(grid[0, 1])
    corr_ax.scatter(
        test_df["Actuals"], test_df["Predictions"], label="Predictions", s=0.5
    )
    corr_ax.scatter(test_df["Actuals"], zero_series_test, label="Naive", s=0.5)
    corr_ax.plot(
        [test_df["Actuals"].min(), test_df["Actuals"].max()],
        [test_df["Actuals"].min(), test_df["Actuals"].max()],
        "r--",
        label="Perfect Prediction",
        linewidth=0.5,
    )
    corr_ax.set_xlabel("Actual Values")
    corr_ax.set_ylabel("Predicted Values")
    corr_ax.set_title("Actual vs Predicted Values")
    corr_ax.legend()

    timeseries_ax = plt.subplot(grid[1, 0])
    timeseries_ax.plot(test_df.index, residuals, label="Residuals")
    timeseries_ax.set_title("Residuals Time Series")
    timeseries_ax.set_xlabel("Date")
    timeseries_ax.set_ylabel("Residuals")
    timeseries_ax.grid(True, alpha=0.2)

    # Histogram with normal distribution fit
    hist_ax = plt.subplot(grid[1, 1])
    sns.histplot(residuals, kde=True, ax=hist_ax, stat="density")
    xmin, xmax = hist_ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    hist_ax.plot(x, p, "k", linewidth=2)
    hist_ax.set_title("Distribution of residuals")
    hist_ax.set_xlabel("Residuals")
    hist_ax.set_ylabel("Density")

    # ACF Plot
    acf_ax = plt.subplot(grid[0, 2])
    plot_acf(
        residuals, ax=acf_ax, alpha=0.05, title="ACF Plot of Reisduals", zero=False
    )
    acf_ax.set_xlabel("Lag")
    acf_ax.set_ylabel("ACF")
    acf_ax.grid(True, alpha=0.2)

    grid.tight_layout(fig)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the best model of a given experiment"
    )
    parser.add_argument(
        "-n", "--name", help="name of the experiment to evaluate", required=True
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

    main(args.name, args.trial_num, args.best)

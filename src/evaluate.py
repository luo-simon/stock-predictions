from unittest import result
import optuna
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import scipy.stats as stats

from lightning.pytorch.trainer.trainer import Trainer

from src.models.CNN.model import CNNModel
from src.models.CNN.data import CNNDataModule
from src.misc import load_processed_dataset

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)


def compute_metrics(predicted, actuals, verbose=False):
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


def backtest(preds_series, actuals_series, buy_and_hold=False, verbose=False):
    action = np.where(preds_series <= 0, "hold", "buy")
    cumulative_return = 1 * np.exp(
        (actuals_series * np.where(action == "hold", 0, 1)).cumsum()
    )
    if buy_and_hold:
        cumulative_return = 1 * np.exp(actuals_series.cumsum())
    if verbose:
        print(
            f"Cumulative returns: {cumulative_return.iloc[-1]}. Standard deviation: {cumulative_return.std()}"
        )
    return cumulative_return.iloc[-1], cumulative_return.std()


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


def predict(trial: optuna.trial.FrozenTrial):
    "Returns a dataframe with predicted and true values for the model of a given Optuna trial"
    ckpt = trial.user_attrs["ckpt_path"]
    data_hparams = trial.user_attrs["data_hparams"]
    stock = data_hparams["stock"]

    data = load_processed_dataset(stock, start_date="2022-01-01", end_date="2024-01-01")
    validation_set = data[:"2023-01-01"]
    test_set = data["2023-01-01":]

    name_map = {"CNN": (CNNModel, CNNDataModule)}

    model = name_map["CNN"][0].load_from_checkpoint(ckpt)
    dm = name_map["CNN"][1](**data_hparams)
    trainer = Trainer()

    validations = trainer.predict(model, dataloaders=dm.val_dataloader())
    validations = (
        torch.cat(validations, dim=0).squeeze(1).cpu().detach().numpy()
    )  # flatten if more than one batch, then squeeze
    validations = pd.Series(validations, index=validation_set.index, name="Predictions")

    predictions = trainer.predict(model, dm)
    predictions = (
        torch.cat(predictions, dim=0).squeeze(1).cpu().detach().numpy()
    )  # flatten if more than one batch, then squeeze
    predictions = pd.Series(predictions, index=test_set.index, name="Predictions")

    val_df = validation_set.join(validations)
    val_df = val_df.rename({"log_return_forecast": "Actuals"}, axis=1)
    test_df = test_set.join(predictions)
    test_df = test_df.rename({"log_return_forecast": "Actuals"}, axis=1)

    return test_df, val_df


def main(experiment_name):
    print(f"Loading {experiment_name}.")
    study = optuna.load_study(
        study_name=experiment_name, storage="sqlite:///optuna_studies.db"
    )
    best = study.best_trial
    print(
        f"Best trial was trial number {best.number} with validation loss of {best.value}. Run completed at {best.datetime_complete}"
    )
    params_str = "".join(f"\n\t- {k}: {v}" for k, v in best.params.items())
    print(f"Sampled parameters were {params_str}")
    user_attrs_str = "".join(f"\n\t- {k}: {v}" for k, v in best.user_attrs.items())
    print(f"User attributes: {user_attrs_str}")

    test_df, val_df = predict(best)

    zero_series_val = pd.Series(index=val_df.index, data=0)
    val_df["Actual Direction"] = np.sign(val_df["Actuals"])
    val_df["Model Direction"] = np.sign(val_df["Predictions"])

    zero_series_test = pd.Series(index=test_df.index, data=0)
    test_df["Actual Direction"] = np.sign(test_df["Actuals"])
    test_df["Model Direction"] = np.sign(test_df["Predictions"])

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

    # One-day ahead metrics
    one_day_metrics = []
    one_day_metrics.append(
        compute_metrics(
            actuals=val_df["Actuals"], predicted=val_df["Predictions"], verbose=False
        )
    )
    one_day_metrics.append(
        compute_metrics(
            actuals=val_df["Actuals"], predicted=zero_series_val, verbose=False
        )
    )
    one_day_metrics.append(
        compute_metrics(
            actuals=test_df["Actuals"], predicted=test_df["Predictions"], verbose=False
        )
    )
    one_day_metrics.append(
        compute_metrics(
            actuals=test_df["Actuals"], predicted=zero_series_test, verbose=False
        )
    )
    one_day_metrics = pd.DataFrame(
        one_day_metrics, columns=["R2", "MSE", "RMSE", "MAE", "p"]
    )
    for key, values in one_day_metrics.items():
        metrics_df[("One-day ahead", key)] = values.values

    index_tuples = [
        ("Validation set", "Model"),
        ("Validation set", "Buy-and-hold"),
        ("Test set", "Model"),
        ("Test set", "Buy-and-hold"),
    ]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["", ""])
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
    accuracies.append(
        (val_df["Actual Direction"] == val_df["Model Direction"]).astype(int).mean()
        * 100
    )
    accuracies.append((val_df["Actual Direction"] == 1).astype(int).mean() * 100)
    accuracies.append(
        (test_df["Actual Direction"] == test_df["Model Direction"]).astype(int).mean()
        * 100
    )
    accuracies.append((test_df["Actual Direction"] == 1).astype(int).mean() * 100)
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
            acc = (df["Actual Direction"] == random_choice).astype(int).mean() * 100
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
    test_df["Actual Close Forecast"] = initial_price * np.exp(
        test_df["Actuals"].cumsum()
    )
    test_df["Close"] = test_df["Actual Close Forecast"].shift(
        1, fill_value=initial_price
    )

    test_df["Predicted Cumsum T+1"] = (
        test_df["Predictions"].rolling(window=2).sum().shift(-1)
    )
    test_df["Predicted Cumsum T+2"] = (
        test_df["Predictions"].rolling(window=3).sum().shift(-2)
    )

    test_df["Predicted Close T+0"] = test_df["Close"] * np.exp(test_df["Predictions"])
    test_df["Predicted Close T+1"] = test_df["Close"] * np.exp(
        test_df["Predicted Cumsum T+1"]
    )
    test_df["Predicted Close T+2"] = test_df["Close"] * np.exp(
        test_df["Predicted Cumsum T+2"]
    )

    prices_ax = plt.subplot(grid[2, :])
    prices_ax.plot(test_df["Actual Close Forecast"], label="Actual Close Forecast")
    prices_ax.plot(test_df["Predicted Close T+0"], label="Predicted Close Forecast")
    prices_ax.plot(test_df["Close"], label="Naive Close Forecast")
    for index, row in test_df.iterrows():
        prices_ax.plot(
            pd.date_range(start=index, periods=3, freq="B"),
            [row[f"Predicted Close T+{i}"] for i in range(3)],
            alpha=0.5,
        )
    prices_ax.set_xlabel("Date")
    prices_ax.set_ylabel("Price")
    prices_ax.legend()
    grid.tight_layout(fig)

    # Residuals
    plt.figure(figsize=(15, 8))
    residuals = test_df["Actuals"] - test_df["Predictions"]
    grid = plt.GridSpec(2, 2)

    preds_ax = plt.subplot(grid[0, 0])
    preds_ax.plot(test_df.index, test_df["Actuals"], label="Actuals")
    preds_ax.plot(test_df.index, test_df["Predictions"], label="Predictions")
    preds_ax.set_xlabel("Date")
    preds_ax.set_ylabel("Values")
    preds_ax.set_title("Actual and Predicted Time Series ")
    preds_ax.legend()

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

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the best model of a given experiment"
    )
    parser.add_argument(
        "-n", "--name", help="name of the experiment to evaluate", required=True
    )
    args = parser.parse_args()

    main(args.name)

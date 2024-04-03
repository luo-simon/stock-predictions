import mlflow
import itertools
import warnings
import random

from src.models.Linear.train import train

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    experiment_id = mlflow.get_experiment_by_name("linear").experiment_id

    feature_combinations = [
        ["Close", "TRANGE"],
        ["Close", "aroon_down"],
        ["Close", "aroon_up"],
        ["Close", "dayofweek"],
        ["Close", "dayofmonth", "williams_r"],
        ["Close", "EMA_50", "williams_r"],
        ["Close", "close_t-1", "williams_r"],
        ["Close", "close_t-2", "williams_r"],
        ["Close", "close_t-4", "williams_r", "TRANGE", "aroon_up"],
        ["Close", "close_t-4", "williams_r", "TRANGE", "aroon_down"],
        ["Close", "close_t-4", "SMA_10"],
        ["Close", "close_t-5", "williams_r"],
        ["Close", "close_t-1", "close_t-2", "close_t-3", "williams_r"],
    ]

    poss = [
        "Open",
        "High",
        "Low",
        "Volume",
        "Dividends",
        "Stock Splits",
        "log_open",
        "log_high",
        "log_low",
        "log_close",
        "log_volume",
        "close_t-1",
        "close_t-2",
        "close_t-3",
        "close_t-4",
        "close_t-5",
        "pct_change",
        "log_return",
        "dayofweek",
        "quarter",
        "month",
        "year",
        "dayofyear",
        "dayofmonth",
        "weekofyear",
        "upper_band",
        "middle_band",
        "lower_band",
        "SMA_3",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_3",
        "EMA_5",
        "EMA_10",
        "EMA_20",
        "EMA_50",
        "ADX",
        "aroon_down",
        "aroon_up",
        "macd",
        "macdsignal",
        "macdhist",
        "RSI_14",
        "slow_k",
        "slow_d",
        "AD",
        "OBV",
        "NATR",
        "fed_funds_rate",
        "log_fed_funds_rate",
        "^N225",
        "log_^N225",
        "^IXIC",
        "log_^IXIC",
        "^FTSE",
        "log_^FTSE",
        "^SPX",
        "log_^SPX",
        "^DJI",
        "log_^DJI",
    ]

    feature_combinations = list(list(x) for x in itertools.combinations(poss, 1))

    while True:
        fs = sorted(random.choices(poss, k=6))
        features = ["Close", "SMA_5", "williams_r", "TRANGE"] + fs
        # Search for runs with the same parameters in this experiment
        query = f'params.features="{features}"'
        existing_runs = mlflow.search_runs(
            experiment_ids=[experiment_id], filter_string=query
        )
        if not existing_runs.empty:
            print(f"features={features} has already been run.")
            continue
        print(f"Running features={features}.")
        train(features)

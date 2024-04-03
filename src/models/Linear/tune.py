import mlflow
import itertools
import warnings
import random

from src.models.Linear.train import train

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    experiment_id = mlflow.get_experiment_by_name("linear").experiment_id

    poss = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits',
    'close_t-1', 'close_t-2', 'close_t-3', 'close_t-4',
       'close_t-5', 'pct_change', 'return', 'dayofweek', 'quarter', 'month',
       'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'upper_band',
       'middle_band', 'lower_band', 'SMA_3', 'SMA_5', 'SMA_10', 'SMA_20',
       'SMA_50', 'EMA_3', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'ADX',
       'aroon_down', 'aroon_up', 'macd', 'macdsignal', 'macdhist', 'RSI_14',
       'slow_k', 'slow_d', 'williams_r', 'AD', 'OBV', 'NATR', 'TRANGE',
       'fed_funds_rate', '^N225', '^IXIC', '^FTSE', '^SPX', '^DJI']


    while True:
        fs = sorted(random.sample(poss, k=7))
        features = ["Close"] + fs
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

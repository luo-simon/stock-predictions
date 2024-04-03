import argparse
import itertools
import yaml
from src.models.ARIMA.train import train
import mlflow
import random
import pandas as pd

if __name__ == "__main__":
    # Load the configuration file:
    with open("configs/arima.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Compute all possible combinations:
    params = config["hyperparameter_tuning"]
    experiment = mlflow.get_experiment_by_name("arima_tune")
    keys, values = zip(*params.items())
    runs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for run in runs:
        existing_runs = pd.DataFrame()
        if experiment:
            # Search for runs with the same parameters in this experiment
            query = " and ".join(
                [f"params.{param}='{value}'" for param, value in run.items()]
            )
            existing_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id], filter_string=query
            )

        if not existing_runs.empty:
            print(f"{run} has already been run.")
            continue

        print(f"Running with {run}.")
        train(run["p"], run["d"], run["q"], "arima_tune")

import itertools
import yaml
import mlflow
import random
import pandas as pd

from src.models.Transformer.train import train


if __name__ == "__main__":
    with open("configs/transformer.yaml", "r") as file:
        config = yaml.safe_load(file)

    params = config["hyperparameter_tuning"]
    experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])
    keys, values = zip(*params.items())
    runs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    random.shuffle(runs)

    for run in runs:
        existing_runs = pd.DataFrame()
        if experiment:
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
        train(
            config["data"]["features"],
            run["sequence_len"],
            run["num_layers"],
            run["dropout"],
            run["num_epochs"],
            run["lr"],
            config["mlflow"]["experiment_name"],
        )

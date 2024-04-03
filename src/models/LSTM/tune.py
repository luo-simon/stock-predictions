import argparse
import itertools
import yaml
from src.models.LSTM.train import train
import mlflow
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning of LSTM model")
    parser.add_argument("--config-file", "-c", type=str, default="configs/lstm.yaml")
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Compute all possible combinations:
    params = config["hyperparameter_tuning"]

    experiment_id = mlflow.get_experiment_by_name("lstm_tuning").experiment_id

    keys, values = zip(*params.items())
    runs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    random.shuffle(runs)

    # Train:
    for run in runs:
        # Search for runs with the same parameters in this experiment
        query =" and ".join([f"params.{param}='{value}'" for param, value in run.items()])
        existing_runs = mlflow.search_runs(
            experiment_ids=[experiment_id], filter_string=query
        )
        if not existing_runs.empty:
            print(f"{run} has already been run.")
            continue
        print(f"Running with {run}.")

        train(
            config["data"]["features"],
            run["sequence_len"],
            run["hidden_dim"],
            run["num_layers"],
            run["num_epochs"],
            run["batch_size"],
            run["lr"],
            config["training"]["output_path"],
            config["mlflow"]["tracking_uri"],
            config["mlflow"]["experiment_name"],
            config["mlflow"]["tags"],
        )

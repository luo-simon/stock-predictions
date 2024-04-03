import argparse
import itertools
import yaml
from src.models.LSTM.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning of LSTM model")
    parser.add_argument("--config-file", "-c", type=str, default="configs/lstm.yaml")
    args = parser.parse_args()

    # Load the configuration file:
    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    # Compute all possible combinations:
    params = config["hyperparameter_tuning"]
    keys, values = zip(*params.items())
    runs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Train:
    for run in runs:
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

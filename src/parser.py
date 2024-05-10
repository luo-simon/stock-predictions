import argparse
from jsonargparse import ArgumentParser, ActionConfigFile


def get_parser(default_config=""):
    """
    Gets the common parsers for training models.
    Returns parser, fit subparser and tune subparser.
    """

    parser = ArgumentParser(
        description="Model training", default_config_files=[default_config]
    )
    parser.add_argument(
        "--experiment_name", type=str, help="Experiment name", required=True
    )
    parser.add_argument("--stock", type=str, help="Stock dataset", required=True)
    default_feature_set = [
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
    parser.add_argument(
        "--features",
        nargs="+",
        default=default_feature_set,
        help="List of features to be used",
    )
    parser.add_argument("--config", action=ActionConfigFile)

    parser_fit = ArgumentParser(
        description="Fits the model on manually selected hyperparameters"
    )
    parser_tune = ArgumentParser(
        description="Tunes the model for specified number of trials"
    )
    parser_tune.add_argument("n_trials", type=int, help="Number of trials")

    subcommands = parser.add_subcommands(
        required=True, dest="subcommand", help="Available commands"
    )
    subcommands.add_subcommand("fit", parser_fit)
    subcommands.add_subcommand("tune", parser_tune)

    return parser, parser_fit, parser_tune

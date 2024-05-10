from lightning.pytorch.trainer.trainer import Trainer
from lightning.pytorch.cli import LightningCLI

from src.misc import get_study, update_namespace
import optuna


class TuneTrainer(Trainer):
    """
    Inherits from Lightning PyTorch trainer and adds tune method for custom CLI behaviour
    """

    def tune(self, model, datamodule=None):
        pass


class BaseCLI(LightningCLI):
    """
    Inherits from LightningCLI with override function and custom subcommands (fit/tune)
    """

    def __init__(self, *args, overrides={}, **kwargs):
        self.overrides = overrides
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--experiment_name",
            type=str,
            help="Experiment name",
            required=True,
        )
        parser.add_argument(
            "--stock", type=str, help="Stock dataset", required=True
        )
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
        parser.add_argument(
            "--n_trials",
            type=int,
            help="Number of trials to run for tuning",
            default=0,
        )

        parser.link_arguments(
            "experiment_name", "trainer.logger.init_args.name"
        )
        parser.link_arguments("stock", "data.stock")
        parser.link_arguments("features", "data.feature_set")

    def before_instantiate_classes(self):
        self.apply_overrides()

    def apply_overrides(self):
        self.config = update_namespace(self.config, self.overrides)

    def subcommands(self):
        """
        Overrides LightningCLI.subcommands() : defines the list of available subcommands and the arguments to skip.
        """
        return {
            "fit": {
                "model",
                "train_dataloaders",
                "val_dataloaders",
                "datamodule",
            },
            "tune": {"model", "dataloaders", "datamodule"},
        }

    def objective(self, trial):
        raise NotImplementedError

    def tune(self, model, datamodule):
        study = get_study(self.config.tune.experiment_name)
        study.optimize(self.objective, n_trials=self.config.tune.n_trials)
        assert (
            len(study.trials) > 0
        ), "No trials found. Run trials first by setting n_trials >= 1."
        best = study.best_trial

        print(
            f"Best trial was trial number {best.number} with validation loss of {best.value}. Run completed at {best.datetime_complete}"
        )
        params_str = "".join(f"\n\t- {k}: {v}" for k, v in best.params.items())
        print(f"Sampled parameters were {params_str}")
        user_attrs_str = "".join(
            f"\n\t- {k}: {v}" for k, v in best.user_attrs.items()
        )
        print(f"User attributes: {user_attrs_str}")

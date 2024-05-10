from re import T
import optuna
from src.models.custom_classes import TuneTrainer, BaseCLI
from src.misc import filter_stdout, get_study

from src.models.LSTM.model import LSTMModel
from src.models.LSTM.data import LSTMDataModule


class LSTMCLI(BaseCLI):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.sequence_len", "model.sequence_len")

    def fit(self, model, datamodule, ckpt_path):
        print(self.config)
        study = get_study(experiment_name=self.config.fit.experiment_name)
        study.enqueue_trial(
            {
                "sequence_len": 5,
                "batch_size": 512,
                "hidden_dim": 4,
                "num_layers": 2,
                "weight_decay": 0,
                "delta": 0.05,
            }
        )
        study.optimize(self.objective, n_trials=1)

    def objective(self, trial):
        # Data hyperparameters
        sequence_len = trial.suggest_categorical("sequence_len", [5, 10, 20, 50])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
        # Model hyperparameters
        hidden_dim = trial.suggest_categorical("hidden_dim", [2, 4, 8])
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 4])
        # Training hyperparameters
        weight_decay = trial.suggest_categorical(
            "weight_decay", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0]
        )
        delta = trial.suggest_categorical("delta", [0.025, 0.05, 0.1, 0.2])
        overrides = {
            "sequence_len": sequence_len,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "weight_decay": weight_decay,
            "delta": delta,
        }

        if hasattr(self.config, "tune"):
            config = self.config.tune
        else:
            config = self.config.fit
            delattr(config, "ckpt_path")

        cli = LSTMCLI(
            LSTMModel,
            LSTMDataModule,
            trainer_class=TuneTrainer,
            run=False,
            args=config,
            overrides=overrides,
        )

        cli.trainer.callbacks.append(
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val_loss"
            )
        )
        cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
        best_checkpoint_path = cli.trainer.checkpoint_callback.best_model_path
        last_checkpoint_path = cli.trainer.checkpoint_callback.last_model_path
        trial.set_user_attr("ckpt_path", best_checkpoint_path)
        trial.set_user_attr("last_ckpt_path", last_checkpoint_path)
        trial.set_user_attr("data_hparams", cli.datamodule.hparams)
        trial.set_user_attr("model", "LSTM")
        return cli.trainer.checkpoint_callback.best_model_score


def main():
    filter_stdout()
    # Run with default trainer configs/trainer_defaults.yaml
    cli = LSTMCLI(LSTMModel, LSTMDataModule, trainer_class=TuneTrainer)


if __name__ == "__main__":
    main()

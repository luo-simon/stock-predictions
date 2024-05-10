import pandas as pd
import numpy as np
import jsonargparse
import optuna
import warnings
import logging


def load_csv_to_df(path):
    """Load a stock historical price csv to a DataFrame with appropriate date index"""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_localize(None)
    return df


def load_processed_dataset(ticker, start_date="2018-01-01", end_date="2023-01-01"):
    df = load_csv_to_df(
        f"/Users/simon/Documents/II/Dissertation/data/processed/{ticker.upper()}.csv"
    )
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    assert (
        start_date >= df.index.min()
    ), f"Start date {start_date} is before the DataFrame's earliest date {df.index.min()}."
    assert (
        end_date <= df.index.max()
    ), f"End date {end_date} is after the DataFrame's latest date {df.index.max()}."
    if start_date > end_date:
        raise ValueError(
            "Start date is after end date after adjustments. No valid data range available."
        )
    return df.loc[start_date:end_date]


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def update_namespace(original, updates):
    """Recursively searches and updates Namespace fields only if they already exist in the original Namespace, regardless of depth.

    :param original: original Namespace
    :type original: Namespace or dict
    :param updates: dictionary with fields and new values
    :type updates: dict
    :return: return updated Namespace
    :rtype: Namespace
    """
    if isinstance(original, jsonargparse.Namespace):
        original = jsonargparse.namespace_to_dict(original)
    assert isinstance(
        original, dict
    ), f"Input not of type Namespace or Dict, but of {type(original)}"
    assert isinstance(updates, dict), f"Updates not Dict, but {updates}"
    for k, v in updates.items():
        if k in original:
            if isinstance(original[k], dict):
                update_namespace(original[k], v)
            else:
                original[k] = v
        else:
            for sub_k, sub_v in original.items():
                if isinstance(sub_v, dict):
                    update_namespace(sub_v, updates)
    return jsonargparse.dict_to_namespace(original)


def filter_stdout():
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings(
        "ignore",
        ".*LightningCLI's args parameter is intended to run from within Python.*",
    )

    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
        logging.WARNING
    )  # Info about GPU/TPU/IPU/HPU


def load_trial_from_experiment(experiment_name, trial_num=None):
    print(f"Loading {experiment_name}.")
    study = optuna.load_study(
        study_name=experiment_name, storage="sqlite:///optuna_studies.db"
    )

    trial = study.best_trial
    print(
        f"Best trial was trial number {trial.number} with validation loss of {trial.value}. Run completed at {trial.datetime_complete}"
    )
    if trial_num:
        trial = study.get_trials()[trial_num]
        print(
            f"Evaluating specified trial number was {trial.number} with validation loss of {trial.value}. Run completed at {trial.datetime_complete}"
        )

    params_str = "".join(f"\n\t- {k}: {v}" for k, v in trial.params.items())
    print(f"Sampled parameters were {params_str}")
    user_attrs_str = "".join(f"\n\t- {k}: {v}" for k, v in trial.user_attrs.items())
    print(f"User attributes: {user_attrs_str}")

    return trial


def load_best_n_trials_from_experiment(experiment_name, n=5):
    print(f"Loading {experiment_name}.")
    study = optuna.load_study(
        study_name=experiment_name, storage="sqlite:///optuna_studies.db"
    )
    all_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    seen = set()
    unique_trials = []
    for trial in all_trials:
        if trial.value and trial.value not in seen:
            seen.add(trial.value)
            unique_trials.append(trial)
    unique_trials = sorted(unique_trials, key=lambda x: x.value, reverse=True)[:n]

    for i, trial in enumerate(unique_trials, 1):
        print(
            f"Rank {i}: trial no. {trial.number}, value: {trial.value}. Run completed at {trial.datetime_complete}"
        )

    return unique_trials


def compute_accuracy(preds, actuals):
    actual_dir = np.sign(actuals)
    preds_dir = np.sign(preds)
    return (actual_dir == preds_dir).mean() * 100.0


def get_study(experiment_name, direction="maximize"):
    study = optuna.create_study(
        study_name=experiment_name,
        storage="sqlite:///optuna_studies.db",
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.NopPruner(),
    )
    return study


def compare_series(s1, s2):
    assert len(s1) == len(s2), f"Length not same: {len(s1)}, {len(s2)}"
    diffs = ~np.isclose(s1, s2, atol=1e-4, rtol=0) & ~(pd.isna(s1) & pd.isna(s2))
    diff_df = pd.DataFrame({"s1": s1, "s2": s2, "diff": diffs})
    assert not diffs.any(), diff_df[diffs]

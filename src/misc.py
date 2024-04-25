import argparse
import jsonargparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# import mlflow
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
    # Check if the start_date is before the first date in the DataFrame
    assert (
        start_date >= df.index.min()
    ), f"Start date {start_date} is before the DataFrame's earliest date {df.index.min()}."
    assert (
        end_date <= df.index.max()
    ), f"End date {end_date} is after the DataFrame's latest date {df.index.max()}."
    # Check if the start date is after the end date after adjustment
    if start_date > end_date:
        raise ValueError(
            "Start date is after end date after adjustments. No valid data range available."
        )
    return df.loc[start_date:end_date]


def create_train_test_set(df, split=0.8, verbose=False):
    features = df.drop(columns=["Close Forecast"], axis=1)
    target = df["Close Forecast"]

    data_len = df.shape[0]

    train_split = int(data_len * split)
    test_split = int(data_len * (1 - split))

    X_train, X_test = features[:train_split], features[train_split:]
    Y_train, Y_test = target[:train_split], target[train_split:]

    if verbose:
        print("Historical Stock Data length is - ", str(data_len))
        print("Training Set length - ", str(train_split))
        print("Test Set length - ", str(test_split))
        print(X_train.shape, X_test.shape)
        print(Y_train.shape, Y_test.shape)

    return X_train, X_test, Y_train, Y_test


def split_data(df, verbose=False):
    """
    Splits dataframe into train, validation and test set on an 80/10/10 ratio
    """
    total_samples = len(df)
    train_split = int(total_samples * 0.8)
    val_split = int(total_samples * 0.9)

    train, val, test = df[:train_split], df[train_split:val_split], df[val_split:]

    if verbose:
        print(f"Original data length: {total_samples}")
        print(f"Train set length: {train_split}")
        print(f"Validation set length: {val_split-train_split}")
        print(f"Test set length: {total_samples-val_split}")
        print(train.shape, val.shape, test.shape)

    return train, val, test


def plot(preds, obs, title=""):
    plt.figure(figsize=(16, 6))
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price", fontsize=18)
    plt.plot(obs, label="Observed")
    plt.plot(preds, label="Predicted")
    plt.grid(visible=True, axis="y")
    plt.title(title)
    plt.legend()
    plt.show()


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(predicted, actuals, verbose=False):
    r2 = r2_score(actuals, predicted)
    mse = mean_squared_error(actuals, predicted)
    rmse = np.sqrt(mean_squared_error(actuals, predicted))
    mae = mean_absolute_error(actuals, predicted)
    mape = get_mape(actuals, predicted)

    if verbose:
        print("R^2: " + str(r2))
        print("MSE: " + str(mse))
        print("RMSE: " + str(rmse))
        print("MAE: " + str(mae))
        print("MAPE: " + str(mape))

    return r2, mse, rmse, mae, mape


def create_sequences(X, y, X_seq_len, y_seq_len, y_include_X=False):
    """
    X : array of feature vectors
    y : array of target values
    X_seq_len : number of days data to include in feature samples (i.e. past N stock price data)
    y_seq_len : number of days data ahead to include in target samples (i.e. next N day forecast)
    y_include_X: should the y sequences include the X samples too - returned y_seq will be length X_seq_len+y_seq_len.

    Returns
        X_seq of shape (num_samples, X_seq_len, num_features)
        y_seq of shape (num_samples, y_seq_len) or (num_samples, y_seq_len+X_seq_len) if y_include_X
    """
    assert len(X) == len(
        y
    ), "Length of feature set and target set not equal, unable to create sequences."
    X_seq = []
    y_seq = []

    for i in range(X_seq_len, len(X) + 2 - y_seq_len):
        X_seq.append(X[i - X_seq_len : i])
        y_start = i - 1
        if y_include_X:
            y_start = y_start - X_seq_len + 1
        y_seq.append(y[y_start : i + y_seq_len - 1])
    return np.array(X_seq), np.array(y_seq).squeeze(2)


def create_sequence(X, sequence_length):
    """
    Given a numpy array, create sequences of a fixed length, where
    each sequence will be used to predict the closing price of the next day.
    """
    out = []
    for i in range(sequence_length, len(X) + 1):
        out.append(X[i - sequence_length : i])
    return np.array(out)


def load_pytorch_model_from_latest_run(experiment_name):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    runs = mlflow.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
        order_by=["start_time desc"],
    )
    assert not runs.empty, "No runs found in specified experiment."
    latest_run_id = runs.iloc[0]["run_id"]
    model = mlflow.pytorch.load_model(f"runs:/{latest_run_id}/models")
    return model


def load_sklearn_model_from_latest_run(experiment_name):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    runs = mlflow.search_runs(
        experiment_ids=[mlflow.get_experiment_by_name(experiment_name).experiment_id],
        order_by=["start_time desc"],
    )
    assert not runs.empty, "No runs found in specified experiment."
    latest_run_id = runs.iloc[0]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
    return model


def load_model_from_experiment(experiment_name, select_by="rmse"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    order_by_mapping = {"recent": "start_time desc", "rmse": "metrics.rmse asc"}

    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=[order_by_mapping[select_by]],
    )

    assert not runs.empty, "No runs found in specified experiment."
    run_id = runs.iloc[0]["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    return model


def load_model_from_run_id(run_id, flavor="pytorch"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    if flavor == "pytorch":
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    elif flavor == "sklearn":
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    else:
        model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    return model


def print_metrics(metrics):
    print("Average Evaluation Metrics")
    print("\tR^2: " + str(metrics[0]))
    print("\tMSE: " + str(metrics[1]))
    print("\tRMSE: " + str(metrics[2]))
    print("\tMAE: " + str(metrics[3]))
    print("\tMAPE: " + str(metrics[4]))


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


def get_one_week_predictions(df):
    """Takes in one-day ahead predictions and computes the one-week ahead predictions

    :param df: DataFrame with Predictions and Actuals columns
    :type df: DataFrame with updated Predictions and Actuals (where each element is list of size 5)
    """
    print(df)

    return df

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
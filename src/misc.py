import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow


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


def plot(preds, obs):
    plt.figure(figsize=(16, 6))
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price", fontsize=18)
    plt.plot(obs, label="Observed")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.show()


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate(predicted, observed, verbose=False):
    r2 = r2_score(observed, predicted)
    mse = mean_squared_error(observed, predicted)
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    mae = mean_absolute_error(observed, predicted)
    mape = get_mape(observed, predicted)

    if verbose:
        print("R^2: " + str(r2))
        print("MSE: " + str(mse))
        print("RMSE: " + str(rmse))
        print("MAE: " + str(mae))
        print("MAPE: " + str(mape))

    return r2, mse, rmse, mae, mape


def create_sequences(Xs, ys, sequence_length):
    """
    Given a numpy array, create sequences of a fixed length, where
    each sequence will be used to predict the closing price of the next day.
    """
    X = []
    y = []
    for i in range(sequence_length, len(Xs) + 1):
        X.append(Xs[i - sequence_length : i])
        y.append(ys[i - 1])
    return np.array(X), np.array(y)


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

    order_by_mapping = {
        "recent": "start_time desc",
        "rmse": "metrics.rmse asc"
    }
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        order_by=[order_by_mapping[select_by]],
    )

    assert not runs.empty, "No runs found in specified experiment."
    run_id = runs.iloc[0]["run_id"]
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    return model

def load_model_from_run_id(run_id):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    return model

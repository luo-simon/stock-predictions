import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_csv_to_df(path):
    """Load a stock historical price csv to a DataFrame with appropriate date index"""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_localize(None)
    return df

def load_processed_dataset(ticker, start_date="2018-01-01", end_date="2023-01-01"):
    df = load_csv_to_df(f"/Users/simon/Documents/II/Dissertation/data/processed/{ticker.upper()}.csv")
    return df.loc[start_date:end_date]

def create_train_test_set(df, split=0.8, verbose=False):
    features = df.drop(columns=['Close Forecast'], axis=1)
    target = df['Close Forecast']

    data_len = df.shape[0]

    train_split = int(data_len * split)
    test_split = int(data_len * (1-split))

    X_train, X_test = features[:train_split], features[train_split:]
    Y_train, Y_test = target[:train_split], target[train_split:]

    if verbose:
        print('Historical Stock Data length is - ', str(data_len))
        print('Training Set length - ', str(train_split))
        print('Test Set length - ', str(test_split))
        print(X_train.shape, X_test.shape)
        print(Y_train.shape, Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test

def plot(preds, obs):
    plt.figure(figsize=(16,6))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
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
        print('MSE: '+str(mse))
        print('RMSE: '+str(rmse))
        print('MAE: '+str(mae))
        print('MAPE: '+str(mape))

    return r2, mse, rmse, mae, mape


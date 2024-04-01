# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

"""
Given the stock price data, with features generated as in the assess stage, 
splits into train test, and returns an evaluation of the model
"""

import numpy as np

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb


# Test for staionarity
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    # Plot rolling statistics:
    plt.plot(timeseries, color="blue", label="Original")
    plt.plot(rolmean, color="red", label="Rolling Mean")
    plt.plot(rolstd, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean and Standard Deviation")
    plt.show(block=False)

    print("Results of dickey fuller test")
    res = adfuller(timeseries, autolag="AIC")
    output = pd.Series(
        res[0:4],
        index=["Test Statistics", "p-value", "# lags used", "# observations used"],
    )
    for key, values in res[4].items():
        output["critical value (%s)" % key] = values

    print(output)

    if res[1] <= 0.05:
        print("Evidence against null hypothesis, so reject. Data is stationary.")
    else:
        print(
            "Weak evidence against null hypothesis, so cannot reject. Data is non-stationary."
        )


def model_1(df, split=0.8):
    """
    Linear regression model with maximum lookback
    """
    X_train, X_test, Y_train, Y_test = create_train_test_set(df, split, verbose=False)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    preds = pd.Series(preds, index=Y_test.index)

    return preds, Y_test


def model_1_1(df, N, split=0.8, verbose=True):
    """
    Given a dataframe, get prediction at each timestep
    Inputs
        df         : dataframe with the values you want to predict
        N          : use previous N values to do prediction
        offset     : for df we only do predictions for df[offset:]
    Outputs
        pred_list  : the predictions for target_col
    """
    X_train, X_test, Y_train, Y_test = create_train_test_set(df, split, verbose=False)
    train_split = int(len(df) * split)

    train = df.drop("Close Forecast", axis=1)

    model = LinearRegression(fit_intercept=True)
    preds = []
    for i in range(train_split, len(df)):
        X_train = train.iloc[i - N : i]
        Y_train = df["Close Forecast"].iloc[i - N : i]
        model.fit(X_train, Y_train)
        pred = model.predict(train.iloc[i : i + 1])
        preds.append(pred[0])
    preds = pd.Series(preds, index=Y_test.index)
    preds[preds < 0] = 0

    return preds, Y_test


def model_2(df, split=0.8):
    """
    ARIMA model
    """
    X_train, X_test, Y_train, Y_test = create_train_test_set(df, split, verbose=False)

    date_index = Y_test.index
    Y_test = Y_test.values
    history = [y for y in Y_train]
    preds = []

    for t in range(len(Y_test)):
        # model_autoARIMA = auto_arima(
        #     X_train,
        #     test='adf',       # use adftest to find optimal 'd'
        #     trace=True,
        # )
        model = ARIMA(history, order=(1, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        preds.append(output[0])
        history.append(Y_test[t])

    preds = pd.Series(preds, index=date_index)
    Y_test = pd.Series(Y_test, index=date_index)
    evaluate(preds, Y_test)
    return preds, Y_test


def model_3(df, split=0.8):
    """
    LSTM network model
    """

    dataset = df.filter(["Close"]).values

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)

    training_data_len = int(np.ceil(len(dataset) * 0.8))
    train_data = scaled_data[0 : int(training_data_len), :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60 : i])
        y_train.append(train_data[i])

    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape, y_train.shape

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 60 :, :]

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60 : i, 0])

    # Convert the data to a numpy array + reshape
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values
    preds = model.predict(x_test)
    preds = scaler.inverse_transform(preds)

    date_index = df.iloc[training_data_len:].index
    preds = pd.Series(preds.flatten(), index=date_index)
    y_test = pd.Series(y_test.flatten(), index=date_index)

    return preds, y_test


def model_4(df, split=0.8):
    """
    XGBoost model
    """

    FEATURES = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "SMA_10",
        "SMA_20",
        "SMA_50",
        "EMA_10",
        "EMA_20",
        "EMA_50",
        "RSI_14",
        "upper_band",
        "middle_band",
        "lower_band",
        "macd",
        "macdsignal",
        "macdhist",
        "OBV",
        "ADX",
        "AD",
        "Aroon_down",
        "Aroon_up",
        "SlowK",
        "SlowD",
        "Williams %R",
        "pct_change",
        "dayofweek",
        "quarter",
        "month",
        "year",
        "dayofyear",
        "dayofmonth",
        "weekofyear",
        "S&P Close",
    ]

    X_train, X_test, Y_train, Y_test = create_train_test_set(df, split)

    X_train = X_train.filter(FEATURES)
    X_test = X_test.filter(FEATURES)

    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        n_estimators=1000,
        early_stopping_rounds=50,
        objective="reg:linear",
        max_depth=3,
        learning_rate=0.01,
    )
    reg.fit(
        X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], verbose=100
    )

    # Feature importance
    fi = pd.DataFrame(
        data=reg.feature_importances_,
        index=reg.feature_names_in_,
        columns=["importance"],
    )
    fi.sort_values("importance").plot(kind="barh", title="Feature Importance")
    plt.show()

    preds = reg.predict(X_test)
    preds = pd.Series(preds, index=Y_test.index)

    return preds, Y_test


def trade(df, preds, verbose=True):
    res_df = df.loc[preds.index[0] : preds.index[-1]][
        ["Open", "Close", "Close Forecast"]
    ]
    res_df["Prediction"] = preds

    pnl = 0

    for i in range(1, len(res_df)):
        today = res_df.iloc[i]
        yesterday = res_df.iloc[i - 1]
        action = ""
        daily_pnl = 0

        # If prediction for Close price at t+1 (today's Close Forecast) is higher than Open at t+1 (today's Open), buy then sell
        if yesterday["Prediction"] > today["Open"]:
            action = "Buy"
            daily_pnl = today["Close"] - today["Open"]  # Buy at Open, sell at Close
            pnl += daily_pnl
        # Else, sell then buy
        elif yesterday["Prediction"] < today["Open"]:
            action = "Sell"
            daily_pnl = today["Open"] - today["Close"]  # Sell at Open, buy at Close
            pnl += daily_pnl

        if verbose:
            print(
                f"{today.name.strftime('%Y-%m-%d'):10s}: {action:4s} @ {today['Open']:8.2f} | Close @ {today['Close']:8.2f} | Day P/L {daily_pnl:8.2f} | Cum. P/L: {pnl:8.2f}"
            )

    return pnl


def trade_buy_hold(data, split=0.8):
    test_split = int(len(data) * (1 - split))
    df = data[test_split:]
    buy = df.iloc[0]["Open"]
    sell = df.iloc[-1]["Close"]
    pnl = sell - buy
    print(f"{df.index[0].strftime('%Y-%m-%d'):10s}: Buy @ {buy:8.2f}")
    print(f"{df.index[-1].strftime('%Y-%m-%d'):10s}: Sell @ {sell:8.2f}")
    print(f"PNL: {pnl}")
    return pnl


def trade_sell_hold(data, split=0.8):
    test_split = int(len(data) * (1 - split))
    df = data[test_split:]
    sell = df.iloc[0]["Open"]
    buy = df.iloc[-1]["Close"]
    pnl = sell - buy
    print(f"{df.index[0].strftime('%Y-%m-%d'):10s}: Buy @ {buy:8.2f}")
    print(f"{df.index[-1].strftime('%Y-%m-%d'):10s}: Sell @ {sell:8.2f}")
    print(f"PNL: {pnl}")
    return pnl
